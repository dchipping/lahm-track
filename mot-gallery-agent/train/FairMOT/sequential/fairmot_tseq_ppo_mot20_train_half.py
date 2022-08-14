import datetime as dt
import os
import os.path as osp
import sys
from pathlib import Path

import gym
from ray import rllib, tune

RUN_NAME = ''
RESULTS_DIR = ''  # tensorboard --logdir $RESULTS_DIR
INITIAL_CHECKPOINT = ''
NUM_LOOPS = 5
NUM_CPUS = 8  # nproc
NUM_GPUS = 1  # nvidia-smi -L | grep GPU | wc -l
STOP_ITERS = 100
CHECKPOINT_FREQ = 25

# Generate test dir and file names
path = Path(__file__)
default_results_dir = sys.argv[1] if len(sys.argv) == 2 else osp.join(
    path.parents[3], "results", path.stem)
results_dir = osp.join(
    RESULTS_DIR, path.stem) if RESULTS_DIR else default_results_dir
run_name = RUN_NAME if RUN_NAME else dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
checkpoint_path = INITIAL_CHECKPOINT if INITIAL_CHECKPOINT else None

# Check env is valid
env = gym.make("motgym:Mot17SequentialEnv-v0")
rllib.utils.check_env(env)

# Default config and stoping criteria, see useful scaling guide:
# https://github.com/ray-project/ray/blob/master/doc/source/rllib/rllib-training.rst#scaling-guide
config = {
    "framework": "torch",
    "num_gpus": NUM_GPUS,
    "num_workers": NUM_CPUS - 1,  # num_workers = Number of similtaneous trials occuring
    "recreate_failed_workers": True,  # For extra stability
    "env": "motgym:Mot17SequentialEnv-v0"
}

stop = {
    "training_iteration": STOP_ITERS,
    # "episode_reward_mean": 90
}

# Run loop n number of times
for _ in range(NUM_LOOPS):
    # Run MOT20-01 training
    mot20_01_config = config.copy()
    mot20_01_config["env"] = "motgym:Mot20SequentialEnvSeq01-v0"
    mot20_01_results = tune.run("PPO",
                                config=mot20_01_config,
                                name=run_name,
                                local_dir=results_dir,
                                stop=stop,
                                restore=checkpoint_path,
                                checkpoint_freq=500,
                                checkpoint_at_end=True)
    checkpoint_path = mot20_01_results.get_last_checkpoint().local_path

    # Run MOT20-02 training
    mot20_02_config = config.copy()
    mot20_02_config["env"] = "motgym:Mot20SequentialEnvSeq02-v0"
    mot20_02_results = tune.run("PPO",
                                config=mot20_02_config,
                                name=run_name,
                                local_dir=results_dir,
                                stop=stop,
                                restore=checkpoint_path,
                                checkpoint_freq=500,
                                checkpoint_at_end=True)
    checkpoint_path = mot20_02_results.get_last_checkpoint().local_path

    # Run MOT20-03 training
    mot20_03_config = config.copy()
    mot20_03_config["env"] = "motgym:Mot20SequentialEnvSeq03-v0"
    mot20_03_results = tune.run("PPO",
                                config=mot20_03_config,
                                name=run_name,
                                local_dir=results_dir,
                                stop=stop,
                                restore=checkpoint_path,
                                checkpoint_freq=500,
                                checkpoint_at_end=True)
    checkpoint_path = mot20_03_results.get_last_checkpoint().local_path

    # Run MOT20-04 training
    mot20_04_config = config.copy()
    mot20_04_config["env"] = "motgym:Mot20SequentialEnvSeq04-v0"
    mot20_04_results = tune.run("PPO",
                                config=mot20_04_config,
                                name=run_name,
                                local_dir=results_dir,
                                stop=stop,
                                restore=checkpoint_path,
                                checkpoint_freq=500,
                                checkpoint_at_end=True)
    checkpoint_path = mot20_04_results.get_last_checkpoint().local_path

# Make checkpoint accessible for inference and benchmarking
src = mot20_04_results.get_last_checkpoint()
dest = osp.join(results_dir, run_name, 'checkpoint')
os.symlink(src, dest)
os.symlink(src + '.tune_metadata', dest + '.tune_metadata')
print(f'Final {path.stem} results saved to: {dest}')
