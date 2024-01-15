import torch
from torch.distributions import MultivariateNormal

from PPOMe.PPOBase import PPOBase


class PPOContinuous(PPOBase):
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
        # Initialize the covariance matrix for the action distribution
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

    def get_action(self, obs, a_memory, c_memory):
        """
        Queries an action from the actor network for continuous action spaces.
        """
        mean, new_a_memory, new_c_memory = super()._get_actor_critic_outputs(
            obs, a_memory, c_memory
        )

        dist = MultivariateNormal(mean, self.cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach().numpy(), log_prob.detach(), new_a_memory, new_c_memory

    def evaluate(self, batch_obs, batch_acts, batch_starts):
        V, means = super().evaluate(batch_obs, batch_starts)
        means = torch.vstack(means)
        dist = MultivariateNormal(means, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        return V, log_probs
