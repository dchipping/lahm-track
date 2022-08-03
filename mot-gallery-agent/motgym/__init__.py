import os
import sys
from gym.envs.registration import register

# Register all envs here for gym to pick-up
register(id='BaseFairmotEnv-v0', entry_point='motgym.envs.FairMOT.base_fairmot_env:BaseFairmotEnv')
register(id='Mot17ParallelEnv-v0', entry_point='motgym.envs.FairMOT.parallel_env:Mot17ParallelEnv')
register(id='Mot20ParallelEnv-v0', entry_point='motgym.envs.FairMOT.parallel_env:Mot20ParallelEnv')
register(id='Mot17SequentialEnvSeq02-v0', entry_point='motgym.envs.FairMOT.sequential_env:Mot17SequentialEnvSeq02')
register(id='Mot17SequentialEnvSeq04-v0', entry_point='motgym.envs.FairMOT.sequential_env:Mot17SequentialEnvSeq04')
register(id='Mot17SequentialEnvSeq05-v0', entry_point='motgym.envs.FairMOT.sequential_env:Mot17SequentialEnvSeq05')
register(id='Mot17SequentialEnvSeq09-v0', entry_point='motgym.envs.FairMOT.sequential_env:Mot17SequentialEnvSeq09')

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = os.path.dirname(__file__)

# Add lib of trackers to PYTHONPATH
lib_path = os.path.join(this_dir, 'trackers')
add_path(lib_path)
