import argparse
import numpy as np
from ray.rllib.utils import check_env
from gym import spaces

import motgym
from motgym.envs.base_env import BasicMotEnv


class BaseJdeEnv(BasicMotEnv):
    def __init__(self, dataset='', detections=''):
        super().__init__(dataset, detections)
        '''
        Action Space: {0, 1}
        0 - Ignore encoding
        1 - Add encoding to gallery
        '''
        self.action_space = spaces.Discrete(2)
        '''
        Observation Space: [1., 1., 100., 1., 1., 1.]
        0. -> 1. - Detection confidence
        0. -> 1. - Max cosine similarity
        0. -> 100. - Feature gallery size
        0. -> 1. - Min IOU Score (Measure of overlap)
        0. -> 1. - Cosine Distance vs. Gallery Average
        0. -> 1. - Avg Cosine Distance Between Tracks
        '''
        self.observation_space = spaces.Box(
            np.zeros((6,)), np.array([1, 1, 5000, 1, 1, 1]), shape=(6,), dtype=float)

        buffer_size = 500  # Tracks are 'never' lost during training!
        parser = argparse.ArgumentParser()
        parser.add_argument('--iou-thres', type=float, default=0.5,
                            help='iou threshold required to qualify as detected')
        parser.add_argument('--conf-thres', type=float,
                            default=0.5, help='object confidence threshold')
        parser.add_argument('--nms-thres', type=float, default=0.4,
                            help='iou threshold for non-maximum suppression')
        parser.add_argument('--min-box-area', type=float,
                            default=200, help='filter out tiny boxes')
        parser.add_argument('--track-buffer', type=int,
                            default=buffer_size, help='tracking buffer')
        opts = parser.parse_args([])
        self.tracker_args = opts

    def reset(self):
        return np.zeros((6,))

    def step(self, action):
        return np.zeros((6,)), 1, True, {}


if __name__ == "__main__":
    env = BaseJdeEnv()
    check_env(env)
