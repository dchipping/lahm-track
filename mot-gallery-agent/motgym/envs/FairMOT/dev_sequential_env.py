'''
Used for *dev* only to test changes to environment and benchmark 
with current train environment.
'''
import random
import numpy as np

from .sequential_env import SequentialFairmotEnv


class DevSequentialFairmotEnv(SequentialFairmotEnv):
    def __init__(self, dataset, detections):
        super().__init__(dataset, detections)

    # def _generate_reward(self):
    #     pass


class Mot17SequentialEnv(DevSequentialFairmotEnv):
    def __init__(self, seq=None):
        dataset = 'MOT17/train_half'
        detections = 'FairMOT/MOT17/train_half'
        super().__init__(dataset, detections)
        if seq: self.seqs = [seq]


class Mot20SequentialEnv(DevSequentialFairmotEnv):
    def __init__(self, seq=None):
        dataset = 'MOT17/train_half'
        detections = 'FairMOT/MOT17/train_half'
        super().__init__(dataset, detections)
        if seq: self.seqs = [seq]


class MotSynthSequentialEnv(DevSequentialFairmotEnv):
    def __init__(self, seq=None):
        dataset = 'MOTSynth/train'
        detections = 'FairMOT/MOTSynth/train'
        super().__init__(dataset, detections)
        if seq: self.seqs = [seq]
