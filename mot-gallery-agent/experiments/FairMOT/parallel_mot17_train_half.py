import gym
from ray import rllib
from ray.rllib.agents import ppo, dqn
from ray.tune.logger import pretty_print

from motgym.envs.FairMOT.parallel_env import Mot17ParallelEnv
# Check env is ok
# env = gym.make("motgym:Mot17ParallelEnv-v0")
class CustomEnv(Mot17ParallelEnv):
    def __init__(self, _):
        super().__init__()
        self.first_render = False
# rllib.utils.check_env(CustomEnv)

# Configure trainer
config = dqn.DEFAULT_CONFIG.copy()
config["framework"] = "torch"
trainer = dqn.DQNTrainer(config=config, env=CustomEnv)

# Train agent on env
for i in range(1000):
    # Perform one iteration of training the policy with PPO
    result = trainer.train()
    print(pretty_print(result))

    if i % 50 == 0:
        checkpoint = trainer.save()
        print("checkpoint saved at", checkpoint)
