import os
import gym

os.unsetenv("SESSION_MANAGER")
env = gym.make("mot_gym:BasicMOT-v1")
obs = env.reset()

done = False
while not done:
    obs, reward, done, info = env.step(1)#env.action_space.sample())
    if info['curr_frame'] % 10 == 0 and info['curr_track']['track_idx'] == 0:
        print(f"Frame: {info['curr_frame']}, TrackIDs: {info['tracks_ids']}")