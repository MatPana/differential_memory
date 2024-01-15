import gym
import numpy as np
from gym.envs.box2d.lunar_lander import LunarLander


class LunarLanderMod(LunarLander):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Removing velocities and momenta from observation space.
        self.subselect = [0, 6, 7]
        # self.subselect = [2, 3, 5, 6, 7]
        self.observation_space = gym.spaces.Box(
            low=np.array([self.observation_space.low[i] for i in self.subselect]),
            high=np.array([self.observation_space.high[i] for i in self.subselect]),
            dtype=np.float32,
        )

    def step(self, action):
        state, reward, done, trun, info = super().step(action)
        partial_state = state[self.subselect]
        # partial_state[0] = abs(partial_state[0])
        return partial_state, reward, done, trun, info

    def reset(self):
        partial_state, i = super().reset()
        return partial_state, i
