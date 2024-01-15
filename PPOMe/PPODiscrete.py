import torch
from torch.distributions import Categorical

from PPOMe.PPOBase import PPOBase


class PPODiscrete(PPOBase):
    def __init__(
        self,
        policy_class,
        critic_class,
        actor_kwargs,
        critic_kwargs,
        env,
        **hyperparameters
    ):
        super().__init__(
            policy_class,
            critic_class,
            actor_kwargs,
            critic_kwargs,
            env,
            **hyperparameters
        )

    def get_action(self, obs, a_memory, c_memory):
        """
        Queries an action from the actor network for discrete action spaces.
        """
        logits, new_a_memory, new_c_memory = super()._get_actor_critic_outputs(
            obs, a_memory, c_memory
        )

        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach().numpy(), log_prob.detach(), new_a_memory, new_c_memory

    def evaluate(self, batch_obs, batch_acts, batch_starts):
        V, logits = super().evaluate(batch_obs, batch_starts)
        logits = torch.vstack(logits)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(batch_acts.squeeze(-1))

        return V, log_probs
