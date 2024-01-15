from os import getpid

from environments.Climbing import Climbing
from environments.Concentration import Concentration
from environments.LunarLanderMod import LunarLanderMod
from networks.ComplexNet import ComplexNet
from networks.DirectMemoryNet import DirectMemoryNet
from networks.GRU import GRUHidden
from utils.utils import *


print(getpid())

log_dir = "logs"

mem_types = ["default", "diag", "sums", "tanh", "mock", "complex", "GRU"]
mem_type = mem_types[1]

# env = LunarLanderMod()
# env = Concentration(5)
env = Climbing()

activation = get_activation(env)

if mem_type == "complex":
    actor_kwargs = {"activation": activation}
    critic_kwargs = {}

    net = ComplexNet

elif mem_type == "GRU":
    actor_kwargs = {"activation": activation}
    critic_kwargs = {}

    net = GRUHidden

else:
    actor_kwargs = {
        "memory_size": 4,
        "hidden_size": 64,
        "memory": mem_type,
        "activation": activation,
    }

    critic_kwargs = {
        "memory_size": 4,
        "hidden_size": 64,
        "memory": mem_type,
    }

    net = DirectMemoryNet

ppo = get_ppo(env)

for i in range(10):
    hyperparameters = {
        "timesteps_per_batch": 2048,
        "max_timesteps_per_episode": 10,
        "gamma": 0.99,
        "n_updates_per_iteration": 10,
        "lr": 3e-3,
        "clip": 0.2,
        "render": False,
        "render_every_i": 10,
        "log_file": f'"{log_dir}/{mem_type}_{i}.out"',
        "device": '"cuda:0"',
    }

    model = ppo(
        policy_class=net,
        critic_class=net,
        actor_kwargs=actor_kwargs,
        critic_kwargs=critic_kwargs,
        env=env,
        **hyperparameters,
    )

    model.learn(total_timesteps=2048 * 128)
