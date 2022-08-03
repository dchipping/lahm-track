import datetime as dt
import os.path as osp
from pathlib import Path

import gym
from ray import rllib, tune

RUN_NAME = ''
RESULTS_DIR = ''

# Generate test dir and file names
path = Path(__file__)
default_results_dir = osp.join(path.parents[2], "results", path.stem)
results_dir = osp.join(
    RESULTS_DIR, path.stem) if RESULTS_DIR else default_results_dir
run_name = RUN_NAME if RUN_NAME else dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
checkpoint_path = None

# Check env is valid
env = gym.make("motgym:Mot17SequentialEnvSeq05-v0")
rllib.utils.check_env(env)

# Default config and stoping criteria, see useful scaling guide:
# https://github.com/ray-project/ray/blob/master/doc/source/rllib/rllib-training.rst#scaling-guide
config = {
    "framework": "torch",
    "num_gpus": 1,
    "num_workers": 7,  # num_workers = Number of similtaneous trials occuring
    "env": "motgym:Mot17SequentialEnvSeq05-v0",
    "recreate_failed_workers": True,  # For extra stability
}

stop = {
    "training_iterations": 10
}

# Run MOT17-05 training
mot17_05_config = config.copy()
mot17_05_config["env"] = "motgym:Mot17SequentialEnvSeq05-v0"
mot17_05_results = tune.run("PPO",
                            config=mot17_05_config,
                            name=run_name,
                            local_dir=results_dir,
                            stop=stop,
                            restore=checkpoint_path,
                            checkpoint_freq=5,
                            checkpoint_at_end=True)
checkpoint_path = osp.join(results_dir, run_name, mot17_05_results.trial_name)

# Run MOT17-02 training
mot17_02_config = config.copy()
mot17_02_config["env"] = "motgym:Mot17SequentialEnvSeq05-v0"
mot17_02_results = tune.run("PPO",
                            config=mot17_02_config,
                            name=run_name,
                            local_dir=results_dir,
                            stop=stop,
                            restore=checkpoint_path,
                            checkpoint_freq=5,
                            checkpoint_at_end=True)
checkpoint_path = osp.join(results_dir, run_name, mot17_02_results.trial_name)
