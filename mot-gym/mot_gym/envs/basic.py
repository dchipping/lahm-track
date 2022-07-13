import os
import os.path as osp
from re import S
import gym
import numpy as np
import torch
import copy
from gym import spaces

import FairMOT.datasets.dataset.jde as datasets
from FairMOT.opts import opts
from FairMOT.tracker.multitracker import JDETracker as Tracker
from FairMOT.tracking_utils.io import read_results


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
        self.observation_space = spaces.Box(0., 1., shape=(3,), dtype=np.float64)

        # Find gym path
        self.gym_path = self._get_gym_path()
        
        # Load seq data and gt
        self.dataloader = None
        self.frame_rate = None
        self.gt_dict = None # ((t, l, w, h), target_id, conf_score)
        self._load_mot17_05()

        # Initialise FairMOT tracker
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        model_path = self._get_model_path('all_dla34.pth')
        opt = opts().init(['mot', f'--load_model={model_path}'])
        self.tracker = Tracker(opt, frame_rate=self.frame_rate)

        # Initialise step variables
        self.frame_id = 0
        self.ep_reward = 0
        self.online_targets = []
        self.prev_online_targets = []
        self.track_idx

    def reset(self):
        self.frame_id = 0
        self.ep_reward = 0
        self.online_targets = []
        self.detections = []
        return 1

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

        # Online Association
        if self.frame_id > 0:
            self.online_targets = self.tracker.associate(self.detections)
        
        # Log results
        # online_tlwhs = []
        # online_ids = []
        # for t in self.online_targets:
        #     tlwh = t.tlwh
        #     tid = t.track_id
        #     vertical = tlwh[2] / tlwh[3] > 1.6
        #     if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
        #         online_tlwhs.append(tlwh)
        #         online_ids.append(tid)
        # self.results.append((frame_id + 1, online_tlwhs, online_ids))

        # Evaluation with gt
        gt_prev = self.gt_dict[self.frame_id-1]
        gt_curr = self.gt_dict[self.frame_id]

        # Calculate reward
        reward = 0

        # Move to next frame and generate detections
        done = False
        if self.track_idx == len(self.online_targets):
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
            self.prev_online_targets = copy.deepcopy(self.online_targets)

            path, img, img0 = self.dataloader[self.frame_id]
            blob = torch.from_numpy(img).cuda().unsqueeze(0)
            self.online_targets = self.tracker.update(blob, img0)
            return done
        else:
            done = True
            return done

    def render(self, mode="human"):
        pass

    def close(self):
        pass

    def _get_obs(self):
        return np.random.random((3,))

    def _get_info(self):
        return {}

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
        self.gt_dict = read_results(osp.join(data_dir, 'gt', 'gt.txt'), 'mot', is_gt=True)

    @staticmethod
    def _get_gym_path():
        gym_dir = osp.dirname(__file__)
        while osp.basename(gym_dir) != 'mot_gym':
            if gym_dir == '/':
                raise Exception('Could not find mot_gym path')
            parent = osp.join(gym_dir, os.pardir)
            gym_dir = os.path.abspath(parent)
        return gym_dir

