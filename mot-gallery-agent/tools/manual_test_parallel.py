import gym


def run_parallel_env(greedy=False):
    env = gym.make("motgym:Mot17ParallelEnv-v0")

    env.reset()
    env.render()

    done = False
    while not done:
        action = 1 if greedy else env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if info['curr_frame'] % 10 == 0 and info['curr_track']['track_idx'] == 0:
            print(f"Frame: {info['curr_frame']}, TrackIDs: {info['tracks_ids']}")
        env.render()
    env.close()
    
    print(info["ep_reward"])


if __name__ == "__main__":
    run_parallel_env()
