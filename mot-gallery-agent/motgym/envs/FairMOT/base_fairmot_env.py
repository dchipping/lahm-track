import numpy as np
from ray.rllib.utils import check_env
from gym import spaces

import motgym
from FairMOT.src.lib.opts import opts
from motgym.envs.base_env import BasicMotEnv


class BaseFairmotEnv(BasicMotEnv):
    def __init__(self, dataset='', detections=''):
        super().__init__(dataset, detections)
        '''
        Action Space: {-1. 0, 1}
        -1 - Remove one of pair of similar feat in gallery
        0 - Ignore encoding
        1 - Add encoding to gallery
        '''
        self.action_space = spaces.Discrete(2)
        '''
        Observation Space: [1., 1., 100., 1., 1., 1.]
        0. -> 1. - Detection confidence
        0. -> 1. - Max cosine similarity
        0. -> 100. - Feature gallery size
        0. -> 1. - Min IOU Score (Mesaure of overlap)
        0. -> 1. - Cosine Distance vs. Gallery Average
        0. -> 1. - Avg Cosine Distnace Between Tracks
        '''
        self.observation_space = spaces.Box(
            np.zeros((6,)), np.array([1, 1, 5000, 1, 1, 1]), shape=(6,), dtype=float)
        
        self.tracker_args = opts().init(['mot'])#, '--batch_size=100'])

    def reset(self):
        return np.zeros((6,))

    def step(self, action):
        return np.zeros((6,)), 1, True, {}

    
if __name__ == "__main__":
    env = BaseFairmotEnv()
    check_env(env)
