import time
from abc import ABC, abstractmethod

import gym
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam


class PPOBase(ABC):
    def __init__(
        self,
        policy_class,
        critic_class,
        actor_kwargs,
        critic_kwargs,
        env,
        **hyperparameters,
    ):
        # Make sure the environment is compatible with our code
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert isinstance(env.action_space, (gym.spaces.Box, gym.spaces.Discrete))

        # Initialize hyperparameters
        self._init_hyperparameters(hyperparameters)

        # Extract environment information
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = (
            env.action_space.shape[0]
            if isinstance(env.action_space, gym.spaces.Box)
            else env.action_space.n
        )
        self.actor_kwargs = actor_kwargs

        # Initialize actor and critic networks
        self.actor = policy_class(self.obs_dim, self.act_dim, **actor_kwargs)
        self.critic = critic_class(self.obs_dim, 1, **critic_kwargs)

        # Initialize optimizers
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        # Logger setup
        self.logger = {
            "delta_t": time.time_ns(),
            "t_so_far": 0,  # Timesteps so far
            "i_so_far": 0,  # Iterations so far
            "batch_lens": [],  # Episodic lengths in batch
            "batch_rews": [],  # Episodic returns in batch
            "actor_losses": [],  # Losses of actor network in current iteration
        }

    @abstractmethod
    def get_action(self, obs, a_memory, c_memory):
        """
        Abstract method to query an action from the actor network.
        Must be implemented by subclasses.
        """
        pass

    def _get_actor_critic_outputs(self, obs, a_memory, c_memory):
        """
        Get outputs from both the actor and critic networks.
        """
        actor_output, new_a_memory = self.actor(obs, a_memory)
        _, new_c_memory = self.critic(obs, c_memory)

        return actor_output, new_a_memory, new_c_memory

    def evaluate(self, batch_obs, batch_starts):
        """
        Estimate the values of each observation and the outputs (means or logits)
        from the actor network.
        """
        new_c_memory = None
        new_a_memory = None

        V = []
        actor_outputs = []

        for obs, b_s in zip(batch_obs, batch_starts):
            if b_s:
                V_single, new_c_memory = self.critic(obs, self.critic.generate_memory())
                actor_output, new_a_memory = self.actor(
                    obs, self.actor.generate_memory()
                )
            else:
                V_single, new_c_memory = self.critic(obs, new_c_memory)
                actor_output, new_a_memory = self.actor(obs, new_a_memory)

            actor_outputs.append(actor_output)
            V.append(V_single)

        return torch.hstack(V), actor_outputs

    def learn(self, total_timesteps):
        """
        Trains the actor and critic networks over a specified number of timesteps.

        Parameters:
            total_timesteps - the total number of timesteps to train for.

        Returns:
            None
        """
        print(
            f"Starting training: {self.max_timesteps_per_episode} timesteps per episode, "
            f"{self.timesteps_per_batch} timesteps per batch, for a total of {total_timesteps} timesteps."
        )

        t_so_far, i_so_far = 0, 0  # Initialize counters for timesteps and iterations

        while t_so_far < total_timesteps:
            batch_data = self.rollout()
            (
                batch_obs,
                batch_acts,
                batch_log_probs,
                batch_rtgs,
                batch_lens,
                batch_starts,
            ) = batch_data
            t_so_far += np.sum(batch_lens)
            i_so_far += 1

            self.logger["t_so_far"], self.logger["i_so_far"] = t_so_far, i_so_far

            V, _ = self.evaluate(
                batch_obs,
                batch_acts,
                batch_starts,
            )

            batch_log_probs = batch_log_probs.to(self.device)
            batch_rtgs = batch_rtgs.to(self.device)
            V = V.to(self.device)

            A_k = batch_rtgs - V.detach()

            A_k = (batch_rtgs - V.detach() - A_k.mean()) / (
                A_k.std() + 1e-10
            )  # Normalized advantage

            for _ in range(self.n_updates_per_iteration):
                V, curr_log_probs = self.evaluate(
                    batch_obs,
                    batch_acts,
                    batch_starts,
                )

                V = V.to(self.device)
                curr_log_probs = curr_log_probs.to(self.device)

                ratios = torch.exp(curr_log_probs - batch_log_probs)
                surr1, surr2 = (
                    ratios * A_k,
                    torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k,
                )
                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs.squeeze(-1))

                # Backpropagation for actor
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                # Backpropagation for critic
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                self.logger["actor_losses"].append(actor_loss.detach())

            self._log_summary()

            if self.save_model and (i_so_far % self.save_freq == 0):
                torch.save(
                    self.actor.state_dict(), f"./model_cache/ppo_actor_{i_so_far}.pt"
                )
                torch.save(
                    self.critic.state_dict(), f"./model_cache/ppo_critic_{i_so_far}.pt"
                )

        print("Training completed.")

    def rollout(self):
        """
        Collect a batch of data from the environment simulations.
        This involves running through the environment using the current policy
        and recording the observations, actions, rewards, and other relevant data.

        Returns:
            batch_obs: Observations collected during the batch. Shape: (number of timesteps, dimension of observation)
            batch_acts: Actions taken during the batch. Shape: (number of timesteps, dimension of action)
            batch_log_probs: Log probabilities of each action taken. Shape: (number of timesteps)
            batch_rtgs: Rewards-To-Go for each timestep. Shape: (number of timesteps)
            batch_lens: Lengths of each episode in the batch. Shape: (number of episodes)
            batch_starts: Flags indicating the start of episodes. Shape: (number of timesteps)
        """
        # Initialize batch data containers
        (
            batch_obs,
            batch_acts,
            batch_log_probs,
            batch_rews,
            batch_lens,
            batch_starts,
        ) = ([], [], [], [], [], [])

        t = 0  # Counter for total timesteps in this batch

        while t < self.timesteps_per_batch:
            ep_rews = []  # Rewards collected per episode
            obs, _ = self.env.reset()  # Reset the environment
            done = False
            a_memory = self.actor.generate_memory()
            c_memory = self.critic.generate_memory()

            for ep_t in range(self.max_timesteps_per_episode):
                # Render the environment at specified intervals
                if (
                    self.render
                    and self.logger["i_so_far"] % self.render_every_i == 0
                    and len(batch_lens) == 0
                ):
                    self.env.render()

                t += 1  # Increment timestep count

                # Record the start of episodes
                batch_starts.append(ep_t == 0)

                # Record observations and memories
                batch_obs.append(obs)

                # Determine action and step through the environment
                action, log_prob, a_memory, c_memory = self.get_action(
                    obs, a_memory, c_memory
                )
                obs, rew, done, _, _ = self.env.step(action)

                # Record action, log probability, and reward
                batch_acts.append(action)
                batch_log_probs.append(log_prob)
                ep_rews.append(rew)

                if done:
                    break

            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)

        # Convert batch data to tensors
        batch_obs = torch.tensor(np.vstack(batch_obs), dtype=torch.float)
        batch_acts = torch.tensor(np.vstack(batch_acts), dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        batch_rtgs = self.compute_rtgs(batch_rews)  # Compute Rewards-To-Go

        # Update logging information
        self.logger["batch_rews"] = batch_rews
        self.logger["batch_lens"] = batch_lens

        return (
            batch_obs,
            batch_acts,
            batch_log_probs,
            batch_rtgs,
            batch_lens,
            batch_starts,
        )

    def compute_rtgs(self, batch_rews):
        """
        Compute Reward-To-Go for each timestep in a batch.
        """
        batch_rtgs = []

        for ep_rews in reversed(batch_rews):
            discounted_reward = 0
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        batch_rtgs = torch.tensor(np.array(batch_rtgs), dtype=torch.float)

        return batch_rtgs

    def _init_hyperparameters(self, hyperparameters):
        """
        Initialize default and custom values for hyperparameters.
        """

        # Algorithm hyperparameters
        self.timesteps_per_batch = 4096
        self.max_timesteps_per_episode = 124
        self.n_updates_per_iteration = 5
        self.lr = 0.005
        self.gamma = 0.95
        self.clip = 0.2

        # Miscellaneous parameters
        self.render = False
        self.render_every_i = 100
        self.save_model = False
        self.save_freq = 10
        self.seed = None
        self.device = "cpu"
        self.log_file = "logs.out"

        # Change any default values to custom values for specified hyperparameters
        for param, val in hyperparameters.items():
            exec("self." + param + " = " + str(val))

        if self.seed != None:
            assert type(self.seed) == int

            torch.manual_seed(self.seed)
            print(f"Successfully set seed to {self.seed}")

    def _log_summary(self):
        """
        Log summary of training progress.
        """

        delta_t = self.logger["delta_t"]
        self.logger["delta_t"] = time.time_ns()
        delta_t = (self.logger["delta_t"] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger["t_so_far"]
        i_so_far = self.logger["i_so_far"]
        avg_ep_lens = np.mean(self.logger["batch_lens"])
        avg_ep_rews = np.mean(
            [np.sum(ep_rews) for ep_rews in self.logger["batch_rews"]]
        )
        avg_actor_loss = np.mean(
            [
                losses.detach().cpu().float().mean()
                for losses in self.logger["actor_losses"]
            ]
        )

        # Round decimal places for more aesthetic logging messages
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))

        # The text you want to append
        text_to_append = f", {avg_ep_rews}" if i_so_far > 0 else f"{avg_ep_rews}"

        # Open the file in 'a' (append) mode
        with open(self.log_file, "a") as file:
            # Write the text to the file
            file.write(text_to_append)

        # Print logging statements
        print(flush=True)
        print(
            f"-------------------- Iteration #{i_so_far} --------------------",
            flush=True,
        )
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Loss: {avg_actor_loss}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        # Reset batch-specific logging data
        self.logger["batch_lens"] = []
        self.logger["batch_rews"] = []
        self.logger["actor_losses"] = []
