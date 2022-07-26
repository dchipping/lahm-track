import os
import sys
from gym.envs.registration import register

# Register all envs here for gym to pick-up
register(id='Mot17Env-v0', entry_point='motgym.envs.FairMOT.parallel_env:Mot17ParallelEnv')
register(id='Mot20Env-v0', entry_point='motgym.envs.FairMOT.parallel_env:Mot20ParallelEnv')
register(id='Mot17Env-v1', entry_point='motgym.envs.FairMOT.sequential_env:Mot17SequentialEnv')

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = os.path.dirname(__file__)

# Add lib of trackers to PYTHONPATH
lib_path = os.path.join(this_dir, 'trackers')
add_path(lib_path)
