import datetime as dt
import os
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
    "recreate_failed_workers": True,  # For extra stability
}

stop = {
    "training_iteration": 10
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
checkpoint_path = mot17_05_results.get_last_checkpoint().local_path

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
checkpoint_path = mot17_02_results.get_last_checkpoint().local_path

# Run MOT17-04 training
mot17_04_config = config.copy()
mot17_04_config["env"] = "motgym:Mot17SequentialEnvSeq04-v0"
mot17_04_results = tune.run("PPO",
                            config=mot17_04_config,
                            name=run_name,
                            local_dir=results_dir,
                            stop=stop,
                            restore=checkpoint_path,
                            checkpoint_freq=5,
                            checkpoint_at_end=True)
checkpoint_path = mot17_04_results.get_last_checkpoint().local_path

# Run MOT17-09 training
mot17_09_config = config.copy()
mot17_09_config["env"] = "motgym:Mot17SequentialEnvSeq04-v0"
mot17_09_results = tune.run("PPO",
                            config=mot17_09_config,
                            name=run_name,
                            local_dir=results_dir,
                            stop=stop,
                            restore=checkpoint_path,
                            checkpoint_freq=5,
                            checkpoint_at_end=True)

# Make checkpoint accessible for inference and benchmarking
src = mot17_09_results.get_last_checkpoint()
dest = osp.join(results_dir, run_name, 'checkpoint')
os.symlink(src, dest)
os.symlink(src + '.tune_metadata', dest + '.tune_metadata')
print(f'Final MOT17 results saved to: {dest}')
