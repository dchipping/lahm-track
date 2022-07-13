import os
import sys
from gym.envs.registration import register

register(id='BasicMOT-v0', entry_point='mot_gym.envs.basic:BasicMotEnv')
# register(id='MOT17-05-v0',entry_point='mot_gym.envs.mot17.seq05:Mot17Env')

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = os.path.dirname(__file__)

# Add lib of trackers to PYTHONPATH
lib_path = os.path.join(this_dir, 'trackers')
add_path(lib_path)

# trackers = os.listdir(lib_path)
# for tracker in trackers:
#     tracker_path = os.path.join(lib_path, tracker)
#     add_path(tracker_path)
