import os
import sys
from gym.envs.registration import register

# Register Base FairMOT Env
register(id='BaseFairmotEnv-v0',
         entry_point='motgym.envs.FairMOT.base_fairmot_env:BaseFairmotEnv')

# Register Parallel Environments
register(id='Mot17ParallelEnv-v0',
         entry_point='motgym.envs.FairMOT.parallel_env:Mot17ParallelEnv')
register(id='Mot20ParallelEnv-v0',
         entry_point='motgym.envs.FairMOT.parallel_env:Mot20ParallelEnv')
register(id='MotSynthParallelEnv-v0',
         entry_point='motgym.envs.FairMOT.parallel_env:MotSynthParallelEnv')

# Register Sequential Environments
register(id='Mot17SequentialEnv-v0',
         entry_point='motgym.envs.FairMOT.sequential_env:Mot17SequentialEnv')
register(id='Mot20SequentialEnv-v0',
         entry_point='motgym.envs.FairMOT.sequential_env:Mot20SequentialEnv')
register(id='MotSynthSequentialEnv-v0',
         entry_point='motgym.envs.FairMOT.sequential_env:MotSynthSequentialEnv')

# Individual MOT17 Sequential
register(id='Mot17SequentialEnvSeq02-v0',
         entry_point='motgym.envs.FairMOT.variants.individual_sequential_env:Mot17SequentialEnvSeq02')
register(id='Mot17SequentialEnvSeq04-v0',
         entry_point='motgym.envs.FairMOT.variants.individual_sequential_env:Mot17SequentialEnvSeq04')
register(id='Mot17SequentialEnvSeq05-v0',
         entry_point='motgym.envs.FairMOT.variants.individual_sequential_env:Mot17SequentialEnvSeq05')
register(id='Mot17SequentialEnvSeq09-v0',
         entry_point='motgym.envs.FairMOT.variants.individual_sequential_env:Mot17SequentialEnvSeq09')

# Individual MOT20 Sequential
register(id='Mot20SequentialEnvSeq01-v0',
         entry_point='motgym.envs.FairMOT.variants.individual_sequential_env:Mot20SequentialEnvSeq01')
register(id='Mot20SequentialEnvSeq02-v0',
         entry_point='motgym.envs.FairMOT.variants.individual_sequential_env:Mot20SequentialEnvSeq02')
register(id='Mot20SequentialEnvSeq03-v0',
         entry_point='motgym.envs.FairMOT.variants.individual_sequential_env:Mot20SequentialEnvSeq03')
register(id='Mot20SequentialEnvSeq04-v0',
         entry_point='motgym.envs.FairMOT.variants.individual_sequential_env:Mot20SequentialEnvSeq04')


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = os.path.dirname(__file__)

# Add lib of trackers to PYTHONPATH
lib_path = os.path.join(this_dir, 'trackers')
add_path(lib_path)
