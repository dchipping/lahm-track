from collections import defaultdict
import random
import gym
import time

# Envs
from motgym.envs.FairMOT.dev_sequential_env import *


def run_sequential_env(greedy=False, target_idx=None, seq=None):
    env = Mot20SequentialEnv(seq=seq)
    env.reset()
    env.render()

    # Option to fix target for repeat comparison
    # if target_idx:
    #     env.assign_target(target_idx)
    print(env.viable_tids, len(env.viable_tids))

    done = False
    while not done:
        action = 1 if greedy else env.action_space.sample()
        obs, reward, done, info = env.step(action)

        if info['curr_frame'] % 10 == 0:
            print(
                f"Frame: {info['curr_frame']}, TrackIDs: {info['curr_track']}")

        env.render()
    env.close()

    print(info["ep_reward"])
    return info["ep_reward"]


if __name__ == "__main__":
    # run_sequential_env()
    run_sequential_env(target_idx=48, seq='MOT17-04')

    # Variation test
    # results = defaultdict(set)
    # for i in range(1, 11):
    #     for n in range(3):
    #         ep_reward = run_sequential_env(target_idx=i, seq='MOT17-04')
    #         results[i].add(ep_reward)
    # print(results.items())
