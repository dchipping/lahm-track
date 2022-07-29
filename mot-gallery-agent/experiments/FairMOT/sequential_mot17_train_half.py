import gym
from ray import rllib, tune
from ray.tune.registry import register_env
from ray.rllib.agents import ppo, dqn
from ray.tune.logger import pretty_print
from ray.rllib.env.multi_agent_env import make_multi_agent


# Check env is ok
env = gym.make("motgym:Mot17SequentialEnvSeq05-v0")
rllib.utils.check_env(env)

results = tune.run("PPO",
                config={
                    "framework": "torch",
                    "num_workers": 1,
                    "num_gpus": 1,
                    "env":"motgym:Mot17SequentialEnvSeq05-v0",
                },
                stop={
                    "training_iteration": 10
                },
                checkpoint_freq=5,
                checkpoint_at_end=True)

# Configure trainer
config = ppo.DEFAULT_CONFIG.copy()
config["framework"] = "torch"
trainer = ppo.DQNTrainer(config=config, env="motgym:Mot17SequentialEnvSeq05-v0")

# Train agent on env
for i in range(1000):
    # Perform one iteration of training the policy with PPO
    result = trainer.train()
    print(pretty_print(result))

    if i % 50 == 0:
        checkpoint = trainer.save()
        print("checkpoint saved at", checkpoint)
