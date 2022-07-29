import gym
from ray import rllib
from ray.tune.registry import register_env
from ray.rllib.agents import ppo, dqn
from ray.tune.logger import pretty_print
from ray.rllib.env.multi_agent_env import make_multi_agent
from motgym.envs.FairMOT.sequential_env import Mot17SequentialEnvSeq05

# Check env is ok
ma_env_cls = make_multi_agent("motgym:Mot17SequentialEnvSeq05-v0")
# rllib.utils.check_env(ma_env_cls())

# Register multi agent class
register_env("multi_agent_mot", lambda _: ma_env_cls({"num_agents": 4}))

# Configure trainer
config = ppo.DEFAULT_CONFIG.copy()
config["framework"] = "torch"
config["num_workers"] = 1
# config["monitor"] = True
trainer = ppo.PPOTrainer(config=config, env="multi_agent_mot") 

# Train agent on env
for i in range(10):
    # Perform one iteration of training the policy with PPO
    result = trainer.train()
    print(pretty_print(result))

    if i % 50 == 0:
        checkpoint = trainer.save()
        print("checkpoint saved at", checkpoint)
