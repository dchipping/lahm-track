import gym
import time


def run_sequential_env(greedy=False, target_idx=None, analyse=False):
    env = gym.make("motgym:Mot17SequentialEnvSeq05-v0")

    # Option to fix target for repeat comparison
    if target_idx: env.assign_target(target_idx)

    env.reset()
    env.render()

    done = False
    while not done:
        action = 1 if greedy else env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(reward)
        # if info['curr_frame'] % 10 == 0:
            # print(f"Frame: {info['curr_frame']}, TrackIDs: {info['curr_track']}")
        # Artificially slow down the rendering for anlaysis
        time.sleep(0.1)
        env.render()
        
    env.close()
    print(info["ep_reward"])


if __name__ == "__main__":
    # run_sequential_env(target_idx=18, greedy=True)
    # run_sequential_env(target_idx=40)
    run_sequential_env(target_idx=20)
