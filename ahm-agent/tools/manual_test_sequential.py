from collections import defaultdict
import random
import gym
import time
from ray.rllib.agents import dqn, impala, ppo

# Envs
from motgym.envs.FairMOT.dev_sequential_env import *


def run_sequential_env(greedy=False, target_idx=None, seq=None):
    # env = Mot17SequentialEnv()
    env = MotSynthSequentialEnv()
    # env = Mot20SequentialEnv(seq='MOT17-1')
    obs = env.reset()
    env.render()
    time.sleep(1)

    # config = ppo.DEFAULT_CONFIG.copy()
    # config["framework"] = "torch"
    # trainer = ppo.PPOTrainer(
    #     config=config, env="motgym:BaseFairmotEnv-v0")
    # trainer.restore(
    #     '/home/dchipping/project/dan-track/mot-gallery-agent/results/fairmot_seq_ppo_mot17_train_half/2022-08-11T05-04-20/checkpoint')

    # Option to fix target for repeat comparison
    # if target_idx:
    #     env.assign_target(target_idx)
    print(env.viable_tids, len(env.viable_tids))

    done = False
    while not done:
        # action = trainer.compute_single_action(obs)
        action = 1  # if greedy else env.action_space.sample()
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
    run_sequential_env(target_idx=7)

    # Variation test
    # results = defaultdict(set)
    # for i in range(1, 11):
    #     for n in range(3):
    #         ep_reward = run_sequential_env(target_idx=i, seq='MOT17-04')
    #         results[i].add(ep_reward)
    # print(results.items())
