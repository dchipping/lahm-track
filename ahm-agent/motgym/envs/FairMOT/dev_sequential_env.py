'''
Used for *dev* only to test changes to environment and benchmark 
with current train environment.
'''
import random
import numpy as np
import os.path as osp

from .sequential_env import SequentialFairmotEnv


class DevSequentialFairmotEnv(SequentialFairmotEnv):
    def __init__(self, dataset, detections, seq=None):
        super().__init__(dataset, detections)
        if seq:
            self.seq = seq
            data_dir = osp.join(self.data_dir, self.seq)
            print(f'Overriding data loader from: {data_dir}')
            self._load_dataset(self.seq)
            self._load_detections(self.seq)

    # def _generate_reward(self):
    #     pass


class Mot17SequentialEnv(DevSequentialFairmotEnv):
    def __init__(self, seq=None):
        dataset = 'MOT17/train_half'
        detections = 'FairMOT/MOT17/train_half'
        super().__init__(dataset, detections, seq)


class Mot20SequentialEnv(DevSequentialFairmotEnv):
    def __init__(self, seq=None):
        dataset = 'MOT17/val_half'
        detections = 'FairMOT/MOT17/val_half'
        super().__init__(dataset, detections, seq)


class MotSynthSequentialEnv(DevSequentialFairmotEnv):
    def __init__(self, seq=None):
        dataset = 'MOTSynth/train'
        detections = 'FairMOT/MOTSynth/train'
        super().__init__(dataset, detections, seq)
