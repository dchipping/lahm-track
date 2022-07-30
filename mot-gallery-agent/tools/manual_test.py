import gym
import time
import random

# env = gym.make("motgym:Mot17ParallelEnv-v0")
env = gym.make("motgym:Mot17SequentialEnvSeq05-v0")

obs = env.reset()
env.render()

done = False
while not done:
    obs, reward, done, info = env.step(1)#env.action_space.sample())
    if info['curr_frame'] % 10 == 0:
    # if info['curr_frame'] % 10 == 0 and info['curr_track']['track_idx'] == 0:
        print(f"Frame: {info['curr_frame']}, TrackIDs: {info['tracks_ids']}")
    env.render()
    time.sleep(1)
    
env.close()
print(info["ep_reward"])
