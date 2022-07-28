import gym
from ray import rllib
from ray.rllib.agents import ppo, dqn
from ray.tune.logger import pretty_print
from ray.rllib.env.multi_agent_env import make_multi_agent
from motgym.envs.FairMOT.sequential_env import Mot17SequentialEnvSeq05
# Check env is ok
# env = gym.make("motgym:motgym:Mot17SequentialEnvSeq05-v0")
ma_env_cls = make_multi_agent("motgym:Mot17SequentialEnvSeq05-v0")
ma_num = ma_env_cls({"num_agents": 2})
print(ma_num.reset())

MultiAgentCustomRenderedEnv = make_multi_agent(
    lambda config: Mot17SequentialEnvSeq05(config))

# Configure trainer
config = dqn.DEFAULT_CONFIG.copy()
config["num_agents"] = 2
config["framework"] = "torch"
trainer = dqn.DQNTrainer(config=config, env=ma_env_cls)

# Train agent on env
for i in range(1000):
    # Perform one iteration of training the policy with PPO
    result = trainer.train()
    print(pretty_print(result))

    if i % 50 == 0:
        checkpoint = trainer.save()
        print("checkpoint saved at", checkpoint)
