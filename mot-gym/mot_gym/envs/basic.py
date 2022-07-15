import os
import os.path as osp
import gym
import numpy as np
import torch
import copy
from gym import spaces

import FairMOT.datasets.dataset.jde as datasets
from FairMOT.opts import opts
from FairMOT.modified_FairMOT import ModifiedJDETracker as Tracker
from FairMOT.tracking_utils.evaluation import Evaluator
from FairMOT.tracking_utils.io import unzip_objs


class BasicMotEnv(gym.Env):
    def __init__(self):
        '''
        Action Space: {0, 1}
        0 - Ignore encoding
        1 - Add encoding to gallery
        '''
        self.action_space = spaces.Discrete(2)
        
        '''
        Observation Space: [1., 1., 1.]
        0->1. - Percentage Overlap IOU of other detections
        0->1. - Detection confidence
        0.->1. - min cosine similarity
        '''
        # self.observation_space = spaces.Box(0., 1., shape=(3,), dtype=float)
        self.observation_space = spaces.Box(0., 1., shape=(2,), dtype=float)

        # Find gym path
        self.gym_path = self._get_gym_path()
        
        # Load seq data and gt
        self.dataloader = None
        self.frame_rate = None
        self.evaluator = None
        self._load_mot17_05()

        # Initialise FairMOT tracker
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        model_path = self._get_model_path('all_dla34.pth')
        self.opt = opts().init(['mot', f'--load_model={model_path}'])
        self.tracker = Tracker(self.opt, self.frame_rate)

    def reset(self):
        self.ep_reward = 0
        self.track_idx = 0
        self.frame_id = 0

        self.online_targets = self._track_update(self.frame_id)
        # Only release once first track(s) has been confirmed
        while not self.online_targets:
            done = self._step_frame()
            if done: raise Exception('Sequence too short')

        track = self.online_targets[self.track_idx]
        obs = self._get_obs(track)
        return obs

    def step(self, action):
        '''
        - Each step a new frame is loaded
        - Detections + encodings generated from FairMOT
        - Each detection + encoding initialised as track class
        - Reassociate previous timestep tracks with current
        - Compare to ground truth, reward for each correct association
        - Eval on MATCH, SWITCH, FP, MISS
        '''
        # Take action
        track = self.online_targets[self.track_idx]
        track.update_gallery(action)

        # Get next targets but freeze track states before so
        # tracker can be restored for all tracks in frame k
        self.frozen_tracks = self._save_tracks()
        next_targets = self._track_update(self.frame_id + 1)
        self._load_tracks(self.frozen_tracks)

        # Compare gt and results for frame k and k+1
        results = {}
        self._add_results(results, self.frame_id, self.online_targets)
        self._add_results(results, self.frame_id + 1, next_targets)
        acc = self._evalute(results)
        mm_type = None

        # Calculate reward
        reward = self._generate_reward(mm_type)

        # Move to next frame and generate detections
        done = False
        if self.track_idx + 1 == len(self.online_targets):
            done = self._step_frame()
            self.track_idx = 0
        else:
            self.track_idx += 1

        # Generate observation and info
        obs = self._get_obs()
        info = self._get_info()

        return obs, reward, done, info

    def _step_frame(self):
        done = False
        if self.frame_id < self.seq_len:
            self.frame_id += 1
            self.online_targets = self._track_update(self.frame_id)
            return done
        else:
            done = True
            return done

    def _track_update(self, frame_id):
        path, img, img0 = self.dataloader[frame_id]
        blob = torch.from_numpy(img).cuda().unsqueeze(0)
        return self.tracker.update(blob, img0)

    def _save_tracks(self):
        tracks = copy.deepcopy(self.tracker.tracked_stracks)
        lost = copy.deepcopy(self.tracker.lost_stracks)
        removed = copy.deepcopy(self.tracker.removed_stracks)
        return (tracks, lost, removed)

    def _load_tracks(self, frozen_tracks):
        tracks, lost, removed = frozen_tracks
        self.tracker.tracked_stracks = tracks
        self.tracker.lost_stracks = lost
        self.tracker.removed_stracks = removed

    def _add_results(self, results, frame_id, online_targets):
        results.setdefault(frame_id, [])
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            ts = t.score
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > self.opt.min_box_area and not vertical:
                track_result = (tuple(tlwh), tid, ts)
                results[frame_id].append(track_result)

    def _evalute(self, results):
        self.evaluator.reset_accumulator()

        frames = sorted(list(set(self.gt_frame_dict.keys()) | set(results.keys())))
        for frame_id in frames:
            trk_objs = results.get(frame_id, [])
            trk_tlwhs, trk_ids = unzip_objs(trk_objs)[:2]
            self.evaluator.eval_frame(frame_id, trk_tlwhs, trk_ids, rtn_events=False)

        return self.evaluator.acc

    def _generate_reward(mm_type):
        '''
         Possible Events: ['RAW', 'FP', 'MISS', 'SWITCH',
          'MATCH', 'TRANSFER', 'ASCEND', 'MIGRATE']
        '''
        match mm_type:
            case 'MATCH':
                return 1
            case 'SWITCH':
                return -1
            case 'FP':
                return -1
            case 'MISS':
                return 0

    # TBD
    # def render(self, mode="human"):
    #     pass

    # TBD
    # def close(self):
    #     pass

    def _get_obs(self, track):
        return track.obs

    def _get_info(self):
        return { "gallery_size": len(self.features) }

    def _get_model_path(self, model_name):
        model_path = osp.join(self.gym_path, 'pretrained', model_name)
        return model_path

    def _load_mot17_05(self, seq_path='MOT17/train/MOT17-05'):
        '''
        MOT submission format:
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
        '''
        data_dir = osp.join(self.gym_path, 'data', seq_path)
        self.dataloader = datasets.LoadImages(osp.join(data_dir, 'img1'))
        meta_info = open(osp.join(data_dir, 'seqinfo.ini')).read()
        self.frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
        self.seq_len = int(meta_info[meta_info.find('seqLen') + 10:meta_info.find('\nimWidth')])
        # self.gt_dict = read_results(osp.join(data_dir, 'gt', 'gt.txt'), 'mot', is_gt=True)
        self.evaluator = Evaluator(osp.join(self.gym_path, 'data'), 'MOT17-05', 'mot')

    @staticmethod
    def _get_gym_path():
        gym_dir = osp.dirname(__file__)
        while osp.basename(gym_dir) != 'mot_gym':
            if gym_dir == '/':
                raise Exception('Could not find mot_gym path')
            parent = osp.join(gym_dir, os.pardir)
            gym_dir = os.path.abspath(parent)
        return gym_dir

