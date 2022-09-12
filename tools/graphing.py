import pandas as pd
import matplotlib.pyplot as plt

# PATH = '/home/dchipping/project/dan-track/ahm-agent/results/policies/train-results/41c8d3f3-c23e/2022-09-09T15-58-23/PPO_motgym:JDE_Mot17SequentialEnv-v0_e0754_00000_0_2022-09-09_15-58-34/progress.csv'
PATH = '/home/dchipping/project/dan-track/ahm-agent/results/a256cb6c-3d5f/2022-09-11T00-05-54/PPO_motgym:JDE_Mot17SequentialEnv-v0_2dd5c_00000_0_2022-09-11_00-06-19/progress.csv'

df = pd.read_csv(PATH)
eps = df['episodes_total']
rwd = df['episode_reward_mean']

with plt.style.context('ggplot'):
    plt.plot(eps, rwd)
    # plt.title('Mean Episode Reward')
    plt.xlabel('Episodes')
    plt.ylabel('Episode Reward')
    plt.show()