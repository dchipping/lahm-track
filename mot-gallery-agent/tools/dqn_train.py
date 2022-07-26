import gym
from ray import rllib
from ray.rllib.agents import ppo, dqn
from ray.tune.logger import pretty_print

# Check env is ok
env = gym.make("motgym:Mot17Env-v0")
rllib.utils.check_env(env)

# Configure trainer
config = dqn.DEFAULT_CONFIG.copy()
config["framework"] = "torch"
trainer = dqn.DQNTrainer(config=config, env="motgym:Mot17Env-v0")

# Train agent on env
for i in range(1000):
   # Perform one iteration of training the policy with PPO
   result = trainer.train()
   print(pretty_print(result))

   if i % 50 == 0:
       checkpoint = trainer.save()
       print("checkpoint saved at", checkpoint)