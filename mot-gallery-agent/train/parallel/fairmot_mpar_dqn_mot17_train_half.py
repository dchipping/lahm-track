import datetime as dt
import os
import os.path as osp
from pathlib import Path

from ray.rllib.env.multi_agent_env import make_multi_agent
from ray.tune.registry import register_env
import gym
from ray import rllib, tune

RUN_NAME = ''
RESULTS_DIR = ''  # tensorboard --logdir $RESULTS_DIR
INITIAL_CHECKPOINT = ''
NUM_LOOPS = 5
NUM_CORES = 7
NUM_GPUS = 1

# Generate test dir and file names
path = Path(__file__)
default_results_dir = osp.join(path.parents[2], "results", path.stem)
results_dir = osp.join(
    RESULTS_DIR, path.stem) if RESULTS_DIR else default_results_dir
run_name = RUN_NAME if RUN_NAME else dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
checkpoint_path = INITIAL_CHECKPOINT if INITIAL_CHECKPOINT else None

# Check env is valid
ma_env_cls = make_multi_agent("motgym:Mot17ParallelEnv-v0")
register_env("multi_agent_parallel_mot17_env",
             lambda _: ma_env_cls({"num_agents": NUM_CORES}))
env = gym.make("multi_agent_parallel_mot17_env")
rllib.utils.check_env(env)

# Default config and stoping criteria, see useful scaling guide:
# https://github.com/ray-project/ray/blob/master/doc/source/rllib/rllib-training.rst#scaling-guide
config = {
    "framework": "torch",
    "num_gpus": NUM_GPUS,  # Important for DQNs
    "recreate_failed_workers": True,  # For extra stability
    "env": "multi_agent_parallel_mot17_env"
}

stop = {
    # "training_iteration": 1000
}

# Run MOT17 training
results = tune.run("DQN",
                   config=config,
                   name=run_name,
                   local_dir=results_dir,
                   stop=stop,
                   restore=checkpoint_path,
                   checkpoint_freq=50,
                   checkpoint_at_end=True)
checkpoint_path = results.get_last_checkpoint().local_path

# Make checkpoint accessible for inference and benchmarking
src = results.get_last_checkpoint()
dest = osp.join(results_dir, run_name, 'checkpoint')
os.symlink(src, dest)
os.symlink(src + '.tune_metadata', dest + '.tune_metadata')
print(f'Final {path.stem} results saved to: {dest}')
