import os
import sys
from gym.envs.registration import register


def register_envs(tracker):
    # Register Parallel Environments
    register(id=f'{tracker}/Mot17ParallelEnv-v0',
             entry_point=f'motgym.envs.{tracker}.parallel_env:Mot17ParallelEnv')
    register(id=f'{tracker}/Mot20ParallelEnv-v0',
             entry_point=f'motgym.envs.{tracker}.parallel_env:Mot20ParallelEnv')
    register(id=f'{tracker}/MotSynthParallelEnv-v0',
             entry_point=f'motgym.envs.{tracker}.parallel_env:MotSynthParallelEnv')

    # Register Sequential Environments
    register(id=f'{tracker}/Mot17SequentialEnv-v0',
             entry_point=f'motgym.envs.{tracker}.sequential_env:Mot17SequentialEnv')
    register(id=f'{tracker}/Mot20SequentialEnv-v0',
             entry_point=f'motgym.envs.{tracker}.sequential_env:Mot20SequentialEnv')
    register(id=f'{tracker}/MotSynthSequentialEnv-v0',
             entry_point=f'motgym.envs.{tracker}.sequential_env:MotSynthSequentialEnv')


# Register Tracker Envs
for tracker in ['FairMOT', 'JDE']:
    register_envs(tracker)
register(id='BaseFairmotEnv-v0',
         entry_point='motgym.envs.FairMOT.base_fairmot_env:BaseFairmotEnv')
register(id='BaseJdeEnv-v0',
         entry_point='motgym.envs.JDE.base_jde_env:BaseJdeEnv')


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


# Add lib of trackers to PYTHONPATH
this_dir = os.path.dirname(__file__)
lib_path = os.path.join(this_dir, 'trackers')
add_path(lib_path)


# Individual MOT17 Sequential
# register(id='Mot17SequentialEnvSeq02-v0',
#          entry_point='motgym.envs.FairMOT.variants.individual_sequential_envs:Mot17SequentialEnvSeq02')
# register(id='Mot17SequentialEnvSeq04-v0',
#          entry_point='motgym.envs.FairMOT.variants.individual_sequential_envs:Mot17SequentialEnvSeq04')
# register(id='Mot17SequentialEnvSeq05-v0',
#          entry_point='motgym.envs.FairMOT.variants.individual_sequential_envs:Mot17SequentialEnvSeq05')
# register(id='Mot17SequentialEnvSeq09-v0',
#          entry_point='motgym.envs.FairMOT.variants.individual_sequential_envs:Mot17SequentialEnvSeq09')

# Individual MOT20 Sequential
# register(id='Mot20SequentialEnvSeq01-v0',
#          entry_point='motgym.envs.FairMOT.variants.individual_sequential_envs:Mot20SequentialEnvSeq01')
# register(id='Mot20SequentialEnvSeq02-v0',
#          entry_point='motgym.envs.FairMOT.variants.individual_sequential_envs:Mot20SequentialEnvSeq02')
# register(id='Mot20SequentialEnvSeq03-v0',
#          entry_point='motgym.envs.FairMOT.variants.individual_sequential_envs:Mot20SequentialEnvSeq03')
# register(id='Mot20SequentialEnvSeq04-v0',
#          entry_point='motgym.envs.FairMOT.variants.individual_sequential_envs:Mot20SequentialEnvSeq04')
