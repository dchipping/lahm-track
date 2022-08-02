import os.path as osp
from collections import defaultdict

import cv2
import random
import numpy as np

import FairMOT.src._init_paths
from modified.fairmot_train import TrainAgentJDETracker as Tracker
from opts import opts

from .base_fairmot_env import BaseFairmotEnv


class SequentialFairmotEnv(BaseFairmotEnv):
    _instance = 0
    def __init__(self, dataset, detections):
        super().__init__(dataset, detections)

    @staticmethod
    def next_instance():
        SequentialFairmotEnv._instance += 1
        return SequentialFairmotEnv._instance
        
    def assign_target(self, track_id=None):
        print(f'Loading data from: {osp.join(self.data_dir, self.seq)}')
        self._load_dataset(self.seq)
        self._load_detections(self.seq)

        gts = self.evaluator.gt_frame_dict.items()
        tid_dict = defaultdict(list)
        for frame_id, gt in gts:
            for tlwh, tid, score in gt:
                tid_dict[tid].append(frame_id)

        viable_tids = [
            tid for tid,
            frame_ids in tid_dict.items() if len(frame_ids) > self.frame_rate * 1]
        if track_id:
            self.focus_tid = viable_tids[track_id]
        else:
            idx = random.randint(0, len(viable_tids)-1)
            print(f'Using random index {idx}')
            self.focus_tid = viable_tids[idx]
            # self.focus_tid = viable_tids[self.next_instance() % len(viable_tids)]
            
        self.frame_ids = tid_dict[self.focus_tid]
        print(f'Assigned ground truth TrackID: {self.focus_tid}')
        print(f'Evaluating frame {self.frame_ids[0]}-{self.frame_ids[-1]} (Len {self.frame_ids[-1]-self.frame_ids[0]})')

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
        if self.frame_id < self.frame_ids[-1]:
            self.frame_id += 1
            self.online_targets = self._track_update(self.frame_id)
            self._save_results(self.frame_id)
        else:
            done = True
        return done

    def _get_obs(self, track):
        return track.obs

    def _get_info(self, track):
        track_info = {
            "track_id": track.track_id,
            "gallery_size": len(track.features),
        }
        seq_info = {
            "seq_len": self.seq_len,
            "frame_rate": self.frame_rate
        }
        return {
            "curr_frame": self.frame_id,
            "ep_reward": self.ep_reward,
            "curr_track": track_info,
            "seq_info": seq_info
        }

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

    def _get_gt_tid(self):
        results = {}
        self._add_results(results, self.frame_id, self.online_targets)

        events = self._get_events(results)
        hypothesis = events[events['OId'] == self.focus_tid]['HId']
        assert len(hypothesis.values) == 1 # Only one HId per frame

        return hypothesis.values[0]

    def _reset_state(self):
        self.frame_id = self.frame_ids[0]
        self.gt_tid = 0
        self.acc_error = 0
        self.tracker = Tracker(self.tracker_args, self.frame_rate)

    def reset(self):
        self._reset_env()
        self._reset_state()

        self.online_targets = self._track_update(self.frame_id)
        # Only release loop once the first track(s) confirmed
        while not self.online_targets or self.gt_tid == 0:
            done = self._step_frame()
            self.gt_tid = self._get_gt_tid()
            if done: raise Exception('Sequence too short')

        self.track = next(filter(lambda x: x.track_id == self.gt_tid,
                     self.online_targets))

        obs = self._get_obs(self.track)
        return obs

    @BaseFairmotEnv.calc_fps
    def step(self, action):
        for track in self.online_targets:
            if self.track != track:
                action = random.randint(0,1)
            track.update_gallery(action, track.curr_feat)

        done = self._step_frame()
        self.gt_tid = self._get_gt_tid()
        TN = not self.gt_tid and not self.track in self.online_targets
        TP = self.track.track_id == self.gt_tid
        if TN or TP:
            reward = 1
            self.acc_error = 1
        else:
            reward = -1 #* self.acc_error
            self.acc_error += 1
        self.ep_reward += reward

        obs = self._get_obs(self.track)
        info = self._get_info(self.track)
        return obs, reward, done, info

    def render(self, mode="human"):
        img0 = cv2.imread(self.images[self.frame_id - 1])
        self._init_rendering(img0.shape)

        # Add bounding box for each track in frame
        for track in self.online_targets:
            tid = track.track_id
            text = str(tid)
            bbox = track.tlwh
            is_correct = (self.gt_tid == tid)
            is_curr_track = (self.track.track_id == tid)
            if is_curr_track and is_correct:
                self._visualize_box(img0, text, bbox, 12, True)
            elif is_correct:
                self._visualize_box(img0, text, bbox, 4, True)
            elif is_curr_track:
                self._visualize_box(img0, text, bbox, 13, True)
            else:
                self._visualize_box(img0, '', bbox, 1, False)

        self._display_frame(img0, self.gt_tid)


class Mot17SequentialEnvSeq02(SequentialFairmotEnv):
    def __init__(self):
        dataset = 'MOT17/train_half'
        detections = 'FairMOT/MOT17/train_half'
        super().__init__(dataset, detections)
        self.seq = 'MOT17-02'
        self.assign_target()


class Mot17SequentialEnvSeq04(SequentialFairmotEnv):
    def __init__(self):
        dataset = 'MOT17/train_half'
        detections = 'FairMOT/MOT17/train_half'
        super().__init__(dataset, detections)
        self.seq = 'MOT17-04'
        self.assign_target()


class Mot17SequentialEnvSeq05(SequentialFairmotEnv):
    def __init__(self):
        dataset = 'MOT17/train_half'
        detections = 'FairMOT/MOT17/train_half'
        super().__init__(dataset, detections)
        self.seq = 'MOT17-05'
        self.assign_target()


class Mot17SequentialEnvSeq09(SequentialFairmotEnv):
    def __init__(self):
        dataset = 'MOT17/train_half'
        detections = 'FairMOT/MOT17/train_half'
        super().__init__(dataset, detections)
        self.seq = 'MOT17-09'
        self.assign_target()
