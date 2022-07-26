import copy
import os.path as osp

import cv2
import numpy as np
from gym import spaces
import FairMOT.src._init_paths
from modified.fairmot_train import ModifiedJDETracker as Tracker
from opts import opts
from tracker.basetrack import BaseTrack

from ..base_env import BasicMotEnv


class ParallelFairmotEnv(BasicMotEnv):
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
        0. -> 1. - Min cosine similarity
        0. -> 100. - Feature gallery size
        '''
        self.observation_space = spaces.Box(
            np.array([0., 0., 0.]), np.array([1., 1., 100.]), shape=(3,), dtype=float)

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
        done = False
        if self.frame_id < self.seq_len:
            self.frame_id += 1
            self.online_targets = self._track_update(self.frame_id)
            self._save_results(self.frame_id)
            return done
        else:
            done = True
            results_file = osp.join(self.results_dir, f'{self.seq}.txt')
            BasicMotEnv._write_results(self.results, results_file, 'mot')
            BasicMotEnv._get_summary(self.evaluator, self.seq, results_file)
            return done

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
        BaseTrack._count  = 0
        self.tracker = Tracker(self.tracker_args, self.frame_rate)

        self.online_targets = self._track_update(self.frame_id)
        # Only release loop once the first track(s) confirmed
        while not self.online_targets:
            done = self._step_frame()
            if done: raise Exception('Sequence too short')

        track = self.online_targets[self.track_idx]
        obs = self._get_obs(track)
        return obs

    def _add_results(self, results_dict, frame_id, online_targets):
        results_dict.setdefault(frame_id, [])
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            ts = t.score
            vertical = tlwh[2] / tlwh[3] > 1.6
            min_area = self.tracker_args.min_box_area
            if tlwh[2] * tlwh[3] > min_area and not vertical:
                track_result = (tuple(tlwh), tid, ts)
                results_dict[frame_id].append(track_result)

    def _evaluate(self, track_id, eval_frame_id): # TODO: Curr limited to k -> k+1
        results = {}
        self._add_results(results, self.frame_id, self.online_targets)

        frozen_count = BaseTrack._count
        forzen_tracks = self.tracker.tracked_stracks
        frozen_lost = self.tracker.lost_stracks
        frozen_removed = self.tracker.removed_stracks
        frozen_kf = self.tracker.kalman_filter

        self.tracker.tracked_stracks = copy.deepcopy(forzen_tracks)
        self.tracker.lost_stracks = copy.deepcopy(frozen_lost)
        self.tracker.removed_stracks = copy.deepcopy(frozen_removed)
        self.tracker.kalman_filter = copy.deepcopy(frozen_kf)

        curr_frame_id = self.frame_id
        while curr_frame_id < eval_frame_id:
            curr_frame_id += 1
            dets = self.detections[str(curr_frame_id)]
            feats = self.features[str(curr_frame_id)]
            online_targets = self.tracker.update(dets, feats, curr_frame_id)
            self._add_results(results, curr_frame_id, online_targets)

        BaseTrack._count = frozen_count
        self.tracker.tracked_stracks = forzen_tracks
        self.tracker.lost_stracks = frozen_lost
        self.tracker.removed_stracks = frozen_removed
        self.tracker.kalman_filter = frozen_kf

        events = self._get_events(results)
        track_events = events[events['HId'] == track_id]
        return track_events['Type'].values

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
        if 'SWITCH' in mm_types:
            return -1000
        else:
            return 1

    @BasicMotEnv.calc_fps
    def step(self, action):
        '''Parallel env flow see env-data-flow.png for design'''
        # Take action
        track = self.online_targets[self.track_idx]
        track.update_gallery(action, track.curr_feat)

        # Look to future to evaluate if successful action
        reward = 0
        if self.frame_id < self.seq_len:
            step = self.frame_rate * 0.2 # How far into future to evaluate
            eval_frame_id = min(self.seq_len - 1, self.frame_id + step)
            mm_types = self._evaluate(track.track_id, eval_frame_id)
            reward = self._generate_reward(track, mm_types)
        self.ep_reward += reward

        # Move to next frame and generate detections
        done = False
        if self.track_idx < len(self.online_targets) - 1:
            self.track_idx += 1
        else:
            done = self._step_frame()
            self.track_idx = 0

        # Generate observation and info for new track
        track = self.online_targets[self.track_idx]
        obs = self._get_obs(track)
        info = self._get_info(track)

        return obs, reward, done, info

    def render(self, mode="human"):
        img0 = cv2.imread(self.images[self.frame_id - 1])
        self._init_rendering(img0.shape)

        # Add bounding box for each track in frame
        for idx in range(len(self.online_targets)):
            track = self.online_targets[idx]
            text = str(track.track_id)
            bbox = track.tlwh
            is_curr_track = (idx == self.track_idx)
            BasicMotEnv._visualize_box(img0, text, bbox, idx, is_curr_track)

        track_id = self.online_targets[self.track_idx].track_id
        self._display_frame(img0, track_id)


class DancetrackParallelEnv(ParallelFairmotEnv):
    def __init__(self):
        dataset = 'dancetrack/train'
        detections = 'FairMOT/dancetrack/train'
        super().__init__(dataset, detections)


class Mot17ParallelEnv(ParallelFairmotEnv):
    def __init__(self):
        dataset = 'MOT17/train_half'
        detections = 'FairMOT/MOT17/train_half'
        super().__init__(dataset, detections)


class Mot20ParallelEnv(ParallelFairmotEnv):
    def __init__(self):
        dataset = 'MOT20/train_half'
        detections = 'FairMOT/MOT20/train_half'
        super().__init__(dataset, detections)


class MotSynthParallelEnv(ParallelFairmotEnv):
    def __init__(self):
        dataset = 'MOTSynth/train'
        detections = 'FairMOT/MOTSynth/train'
        super().__init__(dataset, detections)
