import datetime as dt
import os
import os.path as osp
import sys
from pathlib import Path

import gym
from ray import rllib, tune
from ray.rllib.env.multi_agent_env import make_multi_agent
from ray.tune.registry import register_env

RUN_NAME = ''
RESULTS_DIR = ''  # tensorboard --logdir $RESULTS_DIR
INITIAL_CHECKPOINT = ''
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

# Build multi-agent env
ma_env_cls = make_multi_agent("motgym:Mot17SequentialEnv-v0")
register_env("multi_agent_seq_mot17_env",
             lambda _: ma_env_cls({"num_agents": NUM_CPUS - 1}))

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
    "env": "multi_agent_seq_mot17_env"
}

stop = {
    "training_iteration": STOP_ITERS,
    # "episode_reward_mean": 90
}

# Run MOT17 training
results = tune.run("DQN",
                   config=config,
                   name=run_name,
                   local_dir=results_dir,
                   stop=stop,
                   restore=checkpoint_path,
                   checkpoint_freq=CHECKPOINT_FREQ,
                   checkpoint_at_end=True)
checkpoint_path = results.get_last_checkpoint().local_path

# Make checkpoint accessible for inference and benchmarking
src = results.get_last_checkpoint()
dest = osp.join(results_dir, run_name, 'checkpoint')
os.symlink(src, dest)
os.symlink(src + '.tune_metadata', dest + '.tune_metadata')
print(f'Final {path.stem} results saved to: {dest}')
