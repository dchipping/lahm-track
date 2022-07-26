import gym
import cProfile
from pstats import Stats

env = gym.make("motgym:Mot17Env-v0")
obs = env.reset()

pr = cProfile.Profile()
pr.enable()

done = False
while not done:
    obs, reward, done, info = env.step(1)
    if info['curr_frame'] % 10 == 0 and info['curr_track']['track_idx'] == 0:
        print(f"Frame: {info['curr_frame']}, TrackIDs: {info['tracks_ids']}")

pr.disable()
with open('profiling_stats.txt', 'w') as stream:
    stats = Stats(pr, stream=stream)
    stats.strip_dirs()
    stats.sort_stats('time')
    stats.dump_stats('.prof_stats')
    stats.print_stats()