import gym

env = gym.make("motgym:Mot17Env-v0")
obs = env.reset()
env.render()

done = False
while not done:
    obs, reward, done, info = env.step(env.action_space.sample())
    if info['curr_frame'] % 10 == 0 and info['curr_track']['track_idx'] == 0:
        print(f"Frame: {info['curr_frame']}, TrackIDs: {info['tracks_ids']}")
    env.render()
env.close()
