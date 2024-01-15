import math
from typing import List, Optional

import numpy as np
import torch
from gym import Env, spaces


class Climbing(Env):
    def __init__(
        self, a=0.1, b=0.1, c=0, x_range=(-10, 10), y_range=(-10, 10), **kwargs
    ):
        super().__init__()
        self.a = max(abs(a), 0.1)
        self.b = max(abs(b), 0.1)
        self.c = c
        self.x_range = x_range
        self.y_range = y_range

        # Calculate the minimum height of the bowl
        self.min_height = self.calculate_min_height()

        # Set the success threshold slightly higher than the minimum height
        self.success_threshold = self.min_height + 0.15

        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))
        self.observation_space = spaces.Box(low=0, high=100, shape=(1,))

        self.reset()

    def calculate_min_height(self):
        return self.a * 0**2 + self.b * 0**2 + self.c

    def reset(self, **kwargs):
        r = np.random.uniform(2, 3)
        theta = np.random.uniform(0, 2 * math.pi)

        self.x = r * math.cos(theta)
        self.y = r * math.sin(theta)

        self.present_height = self.height()

        observation = np.array([self.present_height])

        return observation, None

    def height(self):
        return self.a * self.x**2 + self.b * self.y**2 + self.c

    def step(self, action):
        dx = action[0]
        dy = action[1]

        old_height = self.present_height

        self.x = self.x + dx
        self.y = self.y + dy

        self.present_height = self.height()

        reward = old_height - self.present_height  # Reward is the reduction in height

        done = self.present_height < self.success_threshold

        if done:
            reward = 10

        return np.array([self.present_height]), reward, done, None, None
