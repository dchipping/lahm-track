import pprint
import random
import time
from collections import defaultdict

import gym
from motgym.envs.FairMOT.dev_sequential_env import *

labels = ['Detection Confidence',
          'Cosine dist vs. most similar feature',
          'Feature gallery size',
          'IOU score (higher = less overlap)',
          'Cosine dist vs. gallery avg',
          'Avg Cosine dist between all features']


def run_sequential_env(target_idx=None):
    # env = Mot17SequentialEnv('MOT17-02')
    env = gym.make("motgym:Mot17SequentialEnv-v0")

    obs = env.reset()
    env.render()
    
    # Option to fix target for repeat comparison
    if target_idx:
        env.assign_target(target_idx)
    print(env.viable_tids, len(env.viable_tids))


    done = False
    while not done:
        # Inform user of observation
        pprint.pprint(dict(zip(labels, obs)))

        action = None
        while action not in {'0', '1'}:
            action = input("Enter action: ")
        obs, reward, done, info = env.step(int(action))
        print(f'Recieved reward: {reward}')

        if info['curr_frame'] % 10 == 0:
            print(f"Frame: {info['curr_frame']}, \
                 TrackIDs: {info['curr_track']}")
        env.render()

    env.close()
    print(f'Episode reward: {info["ep_reward"]}')


if __name__ == "__main__":
    run_sequential_env()
