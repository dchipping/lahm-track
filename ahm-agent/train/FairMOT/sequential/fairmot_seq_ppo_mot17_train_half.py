import datetime as dt
import os
import os.path as osp
import sys
from pathlib import Path

import gym
import ray
from ray import rllib, tune
from ray.tune import CLIReporter

RUN_NAME = ''
RESULTS_DIR = ''  # tensorboard --logdir $RESULTS_DIR
INITIAL_CHECKPOINT = ''
NUM_CPUS = 8  # nproc
NUM_GPUS = 1  # nvidia-smi -L | grep GPU | wc -l
STOP_ITERS = 100
CHECKPOINT_FREQ = 25
REPORT_FREQ = 900

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

# Startup Ray
ray.shutdown()
ray.init(log_to_driver=False)

# Run MOT17 training
results = tune.run("PPO",
                   config=config,
                   name=run_name,
                   local_dir=results_dir,
                   stop=stop,
                   restore=checkpoint_path,
                   checkpoint_freq=CHECKPOINT_FREQ,
                   checkpoint_at_end=True,
                   progress_reporter=CLIReporter(max_report_frequency=REPORT_FREQ))
checkpoint_path = results.get_last_checkpoint().local_path

# Make checkpoint accessible for inference and benchmarking
src = results.get_last_checkpoint()
dest = osp.join(results_dir, run_name, 'checkpoint')
os.symlink(src, dest)
os.symlink(src + '.tune_metadata', dest + '.tune_metadata')
print(f'Final {path.stem} results saved to: {dest}')
