'''
Used for *dev* only to test changes to environment and benchmark 
with current train environment.
'''
import numpy as np

from .parallel_env import ParallelFairmotEnv


class DevParallelFairmotEnv(ParallelFairmotEnv):
    def __init__(self, dataset, detections):
        super().__init__(dataset, detections)

    def _generate_reward(self, track, mm_types):
        '''
        Each event type is one of the following
        - `'MATCH'` a match between a object and hypothesis was found
        - `'SWITCH'` a match but differs from previous assignment (hypothesisid != previous) (relative to Hypo)
        - `'MISS'` no match for an object was found
        - `'FP'` no match for an hypothesis was found (spurious detections)
        - `'RAW'` events corresponding to raw input
        - `'TRANSFER'` a match but differs from previous assignment (objectid != previous) (relative to Obj)
        - `'ASCEND'` a match but differs from previous assignment  (hypothesisid is new) (relative to Obj)
        - `'MIGRATE'` a match but differs from previous assignment  (objectid is new) (relative to Hypo)
        '''
        reward = 0
        if 'SWITCH' in mm_types:
            reward += -10
        else:
            reward += 1
        # if len(track.features) > 30:
        #     reward += -1
        return reward


class DancetrackParallelEnv(DevParallelFairmotEnv):
    def __init__(self):
        dataset = 'dancetrack/train'
        detections = 'FairMOT/dancetrack/train'
        super().__init__(dataset, detections)


class Mot17ParallelEnv(DevParallelFairmotEnv):
    def __init__(self):
        dataset = 'MOT17/train_half'
        detections = 'FairMOT/MOT17/train_half'
        super().__init__(dataset, detections)


class Mot20ParallelEnv(DevParallelFairmotEnv):
    def __init__(self):
        dataset = 'MOT20/train_half'
        detections = 'FairMOT/MOT20/train_half'
        super().__init__(dataset, detections)


class MotSynthParallelEnv(DevParallelFairmotEnv):
    def __init__(self):
        dataset = 'MOTSynth/train'
        detections = 'FairMOT/MOTSynth/train'
        super().__init__(dataset, detections)
