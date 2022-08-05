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

    def _generate_reward(self):
        TN = not self.gt_tid and not self.track in self.online_targets
        TP = self.track.track_id == self.gt_tid
        prop_reward = 100 / self.seq_len

        if TN or TP:
            reward = prop_reward
            self.acc_error = 1
        else:
            buffer_size = int(self.frame_rate / 30.0 *
                              self.tracker_args.track_buffer)
            if self.acc_error > buffer_size:  # Track permanently lost
                done = True
            # reward = -prop_reward  # * self.acc_error
            reward = 0
            self.acc_error += 1

        return reward, done


class Mot17SequentialEnv(SequentialFairmotEnv):
    def __init__(self):
        dataset = 'MOT17/train_half'
        detections = 'FairMOT/MOT17/train_half'
        super().__init__(dataset, detections)
        self.seq = random.choice(self.seqs)
        self.assign_target()


class Mot20SequentialEnv(SequentialFairmotEnv):
    def __init__(self):
        dataset = 'MOT17/train_half'
        detections = 'FairMOT/MOT17/train_half'
        super().__init__(dataset, detections)
        self.seq = random.choice(self.seqs)
        self.assign_target()


class MotSynthParallelEnv(SequentialFairmotEnv):
    def __init__(self):
        dataset = 'MOTSynth/train'
        detections = 'FairMOT/MOTSynth/train'
        super().__init__(dataset, detections)
        self.seq = random.choice(self.seqs)
        self.assign_target()
