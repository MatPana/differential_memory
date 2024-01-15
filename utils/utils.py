from gym import Env, spaces

from PPOMe.PPOBase import PPOBase
from PPOMe.PPOContinuous import PPOContinuous
from PPOMe.PPODiscrete import PPODiscrete


def get_ppo(env: Env) -> PPOBase:
    if type(env.observation_space) == spaces.Box:
        ppo = PPOContinuous
    else:
        ppo = PPODiscrete
    return ppo


def get_activation(env: Env) -> PPOBase:
    if type(env.observation_space) == spaces.Box:
        activation = "Sig"
    else:
        activation = None
    return activation
