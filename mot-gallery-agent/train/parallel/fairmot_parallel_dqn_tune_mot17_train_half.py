import gym
from ray import rllib
import os
from pathlib import Path
from ray.rllib.agents import ppo, dqn

from ray.tune.logger import pretty_print

from motgym.envs.FairMOT.parallel_env import Mot17ParallelEnv
# Check env is ok
# env = gym.make("motgym:Mot17ParallelEnv-v0")
# class CustomEnv(Mot17ParallelEnv):
#     def __init__(self, _):
#         super().__init__()
#         self.first_render = False
# rllib.utils.check_env(CustomEnv)

# Configure trainer
config = dqn.DEFAULT_CONFIG.copy()
config["framework"] = "torch"
# config["local_dir"] = os.path.join(os.path.dirname(__file__), 'trainresults')
# config["name"] = Path(__file__).name
# trainer = dqn.DQNTrainer(config=config, env=CustomEnv)
trainer = dqn.DQNTrainer(config=config, env="motgym:Mot17ParallelEnv-v0")

# Train agent on env
for i in range(3):
    # Perform one iteration of training the policy with PPO
    result = trainer.train()
    print(pretty_print(result))

    if i % 50 == 0:
        checkpoint = trainer.save()
        print("checkpoint saved at", checkpoint)
