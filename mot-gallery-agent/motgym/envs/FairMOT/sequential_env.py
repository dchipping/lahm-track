import copy
import random
import os.path as osp
from collections import defaultdict

import cv2
import numpy as np
from gym import spaces
import FairMOT.src._init_paths
from modified.fairmot_train import ModifiedJDETracker as Tracker
from opts import opts
from tracker.basetrack import BaseTrack

from ..base_env import BasicMotEnv


class SequentialFairmotEnv(BasicMotEnv):
    def __init__(self, dataset, detections):
        super().__init__(dataset, detections)
        '''
        Action Space: {0, 1}
        0 - Ignore encoding
        1 - Add encoding to gallery
        '''
        self.action_space = spaces.Discrete(2)
        '''
        Observation Space: [1., 1., 100.]
        0. -> 1. - Detection confidence
        0. -> 1. - Max cosine similarity
        0. -> 100. - Feature gallery size
        '''
        self.observation_space = spaces.Box(
            np.array([0., -1., 0.]), np.array([1., 1., 100.]), shape=(3,), dtype=float)

        self.tracker_args = opts().init(['mot'])

    def _track_update(self, frame_id):
        dets = self.detections[str(frame_id)]
        feats = self.features[str(frame_id)]
        return self.tracker.update(dets, feats, frame_id)

    def _save_results(self, frame_id):
        # Filter to only save active tracks
        active_targets = [t for t in self.online_targets if t.is_activated]
        online_ids = []
        online_tlwhs = []
        for t in active_targets:
            tlwh = t.tlwh
            tid = t.track_id
            ts = t.score
            vertical = tlwh[2] / tlwh[3] > 1.6
            min_area = self.tracker_args.min_box_area
            if tlwh[2] * tlwh[3] > min_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
        self.results.append((frame_id, online_tlwhs, online_ids))

    def _step_frame(self):
        pass
        # done = False
        # if self.frame_id < self.seq_len:
        #     self.frame_id += 1
        #     self.online_targets = self._track_update(self.frame_id)
        #     self._save_results(self.frame_id)
        #     return done
        # else:
        #     done = True
        #     results_file = osp.join(self.results_dir, f'{self.seq}.txt')
        #     BasicMotEnv._write_results(self.results, results_file, 'mot')
        #     BasicMotEnv._get_summary(self.evaluator, self.seq, results_file)
        #     return done

    def _get_obs(self, track):
        return track.obs

    def _get_info(self, track):
        tids = {t.track_id for t in self.online_targets}
        track_info = {
            "track_id": track.track_id,
            "gallery_size": len(track.features),
            "track_idx": self.track_idx
        }
        seq_info = {"seq_len": self.seq_len, "frame_rate": self.frame_rate}
        return {
            "curr_frame": self.frame_id,
            "ep_reward": self.ep_reward,
            "tracks_ids": tids,
            "curr_track": track_info,
            "seq_info": seq_info
        }

    def reset(self):
        self._reset_seq()
        self._reset_state()

        gts = self.evaluator.gt_frame_dict.items()
        tid_dict = defaultdict(list)
        for frame_id, gt in gts:
            for tlwh, tid, score in gt:
                tid_dict[tid].append(frame_id)
        viable_tids = [
            tid for tid,
            frame_ids in tid_dict.items() if len(frame_ids) > 30]
        focus_tid = viable_tids[random.randint(0, len(viable_tids))]

        return _

    def step(self, action):
        return _, _, _, _


class Mot17SequentialEnv(SequentialFairmotEnv):
    def __init__(self):
        dataset = 'MOT17/train_half'
        detections = 'FairMOT/MOT17/train_half'
        super().__init__(dataset, detections)
