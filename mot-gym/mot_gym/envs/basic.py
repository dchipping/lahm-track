import os
import os.path as osp
import time
import gym
import numpy as np
import torch
import copy
import datetime as dt
import cv2
import motmetrics as mm
from pathlib import Path
from gym import spaces
from .bbox_colors import _COLORS

import FairMOT.datasets.dataset.jde as datasets
from FairMOT.opts import opts
from FairMOT.modified_FairMOT import ModifiedJDETracker as Tracker
from FairMOT.tracking_utils.evaluation import Evaluator
from FairMOT.tracking_utils.io import unzip_objs
from FairMOT.tracker.basetrack import BaseTrack


# TODO: Why last frame gets dif results from rest?
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
        0.->1. - Min cosine similarity
        '''
        self.observation_space = spaces.Box(0., 1., shape=(2,), dtype=float)

        # Find gym path
        self.gym_path = self._get_gym_path()
        
        # Load seq data and gt
        self.dataloader = None
        self.frame_rate = None
        self.evaluator = None
        self._load_data('short-seq/50-frames')

        # Initialise FairMOT tracker
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        model_path = self._get_model_path('all_dla34.pth')
        self.opt = opts().init(['mot', f'--load_model={model_path}'])
        self.tracker = Tracker(self.opt, self.frame_rate, train_mode=True)

        # Additional variables
        self.first_render = True

    def reset(self):
        self.ep_reward = 0
        self.track_idx = 0
        self.frame_id = 1
        self.results = []

        self.online_targets = self._track_update(self.frame_id)
        # Only release once first track(s) confirmed
        while not self.online_targets:
            done = self._step_frame()
            if done: 
                raise Exception('Sequence too short')

        track = self.online_targets[self.track_idx]
        obs = self._get_obs(track)
        info = self._get_info(track)
        return obs, info

    def step(self, action):
        '''
        See env-data-flow.png for data flow
        '''
        # Take action
        track = self.online_targets[self.track_idx]
        track.update_gallery(action, track.curr_feat)

        # Look to future to evaluate if succesful action
        reward = 0
        if self.frame_id < self.seq_len:
            mm_type = self._evaluate(track.track_id, 1)
            # mm_type = 'LOST'
            reward = self._generate_reward(mm_type)
            # print(mm_type, reward)
        self.ep_reward += reward

        # Move to next frame and generate detections
        done = False
        if self.track_idx + 1 < len(self.online_targets):
            self.track_idx += 1
        else:
            done = self._step_frame()
            self.track_idx = 0

        # Generate observation and info for new track
        track = self.online_targets[self.track_idx]
        obs = self._get_obs(track)
        info = self._get_info(track)

        return obs, reward, done, info

    def _evaluate(self, track_id, eval_step=1): # TODO: Curr limited to k -> k+1
        # Get next targets but freeze track states before so
        # tracker can be restored for all tracks in frame k
        eval_frame_id = self.frame_id + eval_step
        next_targets = self._track_eval(eval_frame_id) 

        # Compare gt and results for frame k and k+1
        results = {}
        self._add_results(results, self.frame_id, self.online_targets)
        self._add_results(results, self.frame_id + 1, next_targets)
        events = self._evalute(results).loc[1]
        
        # Calculate reward
        track_event = events[events['HId'] == track_id]
        mm_type = track_event['Type'].values[0] if track_event.size else 'LOST'
        return mm_type

    def _step_frame(self):
        done = False
        if self.frame_id < self.seq_len:
            self.frame_id += 1
            self.online_targets = self._track_update(self.frame_id)
            self._save_results(self.frame_id, self.online_targets)
            return done
        else:
            done = True
            self._write_results(self.results, 'mot')
            self._get_summray()
            return done

    def _track_update(self, frame_id):
        frame_idx = frame_id - 1
        path, img, img0 = self.dataloader[frame_idx]
        blob = torch.from_numpy(img).cuda().unsqueeze(0)
        if self.frame_id == 50:
            l = 50 % 20
        return self.tracker.update(blob, img0, frame_id)

    def _track_eval(self, eval_frame_id):
        frozen_count = BaseTrack._count
        forzen_tracks = copy.deepcopy(self.tracker.tracked_stracks)
        frozen_lost = copy.deepcopy(self.tracker.lost_stracks)
        frozen_removed = copy.deepcopy(self.tracker.removed_stracks)

        frame_id = self.frame_id
        while frame_id < eval_frame_id:
            frame_id += 1
            frame_idx = frame_id - 1
            path, img, img0 = self.dataloader[frame_idx]
            blob = torch.from_numpy(img).cuda().unsqueeze(0)
            tracks = self.tracker.update(blob, img0, frame_id)

        BaseTrack._count = frozen_count
        self.tracker.tracked_stracks = forzen_tracks
        self.tracker.lost_stracks = frozen_lost
        self.tracker.removed_stracks = frozen_removed

        return tracks

    def _add_results(self, results_dict, frame_id, online_targets):
        results_dict.setdefault(frame_id, [])
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            ts = t.score
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > self.opt.min_box_area and not vertical:
                track_result = (tuple(tlwh), tid, ts)
                results_dict[frame_id].append(track_result)
    
    def _save_results(self, frame_id, online_targets):
        online_ids = []
        online_tlwhs = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            ts = t.score
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > self.opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
        self.results.append((frame_id, online_tlwhs, online_ids))

    def _evalute(self, results): 
        self.evaluator.reset_accumulator()

        frames = sorted(list(set(self.evaluator.gt_frame_dict.keys()) & set(results.keys())))
        for frame_id in frames:
            trk_objs = results.get(frame_id, [])
            trk_tlwhs, trk_ids = unzip_objs(trk_objs)[:2]
            self.evaluator.eval_frame(frame_id, trk_tlwhs, trk_ids, rtn_events=False)

        events = self.evaluator.acc.events
        return events[events['Type'] != 'RAW']

    def _generate_reward(self, mm_type):
        '''
        Possible Events: ['RAW', 'FP', 'MISS', 'SWITCH',
        'MATCH', 'TRANSFER', 'ASCEND', 'MIGRATE']
        '''
        if mm_type == 'MATCH':
            return 1
        elif mm_type == 'SWITCH':
            return -1
        elif mm_type == 'FP':
            return -1
        elif mm_type == 'LOST':
            return 0
        else:       ### TODO: REMOVE AND HANDLE EXTRA MOT METRICS
            return 0 ##

    def render(self, mode="human"):
        path, img, img0 = self.dataloader[self.frame_id - 1]
        if self.first_render:
            black_img = np.zeros(img0.shape, dtype=img0.dtype)
            cv2.imshow('env snapshot', black_img)
            cv2.waitKey(1)
            time.sleep(1)
            self.first_render = False
          
        for i in range(len(self.online_targets)):
            track = self.online_targets[i]
            text = str(track.track_id)
            bbox = track.tlwh
            curr_track = (i == self.track_idx)
            self._visualize_box(img0, text, bbox, i, curr_track)

        curr_tid = self.online_targets[self.track_idx].track_id
        text = f'TrackID {curr_tid}, Frame {self.frame_id}, {self.frame_rate} fps'
        cv2.putText(img0, text, (4,16), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1, cv2.LINE_AA)
        cv2.imshow('env snapshot', img0)
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()
        self.first_render = True

    def _get_obs(self, track):
        return track.obs

    def _get_info(self, track):
        tids = {t.track_id for t in self.online_targets}
        track_info = { "track_id": track.track_id, "gallery_size": len(track.features),
         "track_idx": self.track_idx }
        seq_info = { "seq_len": self.seq_len, "frame_rate": self.frame_rate }
        return { "curr_frame": self.frame_id, "ep_reward": self.ep_reward, 
        "tracks_ids": tids, "curr_track": track_info, "seq_info": seq_info }

    def _get_model_path(self, model_name):
        model_path = osp.join(self.gym_path, 'pretrained', model_name)
        return model_path

    def _load_data(self, seq_path='MOT17/train/MOT17-05'):
        '''
        MOT submission format:
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
        '''
        self.data_dir = osp.join(self.gym_path, 'data', seq_path)
        print(f'Loading data from: {self.data_dir}')
        self.dataloader = datasets.LoadImages(osp.join(self.data_dir, 'img1'))
        meta_info = open(osp.join(self.data_dir, 'seqinfo.ini')).read()
        self.frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
        self.seq_len = int(meta_info[meta_info.find('seqLen') + 10:meta_info.find('\nimWidth')])
        self.evaluator = Evaluator(self.data_dir, '', 'mot')

    def _write_results(self, results, data_type):
        timestamp = dt.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
        results_dir = osp.join(self.data_dir, 'results', f'{timestamp}')
        if not osp.exists(results_dir):
            os.makedirs(results_dir)
        self.results_file = osp.join(results_dir, 'results.txt')
        
        if data_type == 'mot':
            save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
        elif data_type == 'kitti':
            save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
        else:
            raise ValueError(data_type)

        with open(self.results_file, 'w') as f:
            for frame_id, tlwhs, track_ids in results:
                if data_type == 'kitti':
                    frame_id -= 1
                for tlwh, track_id in zip(tlwhs, track_ids):
                    if track_id < 0:
                        continue
                    x1, y1, w, h = tlwh
                    x2, y2 = x1 + w, y1 + h
                    line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                    f.write(line)
        print('save results to {}'.format(self.results_file))

    def _get_summray(self):
        name = Path(self.data_dir).name
        evaluator = Evaluator(self.data_dir, '', 'mot')
        acc = evaluator.eval_file(self.results_file)
        metrics = mm.metrics.motchallenge_metrics
        mh = mm.metrics.create()
        summary = mh.compute(
            acc, 
            metrics=metrics,
            name=name
        )

        strsummary = mm.io.render_summary(
            summary,
            formatters=mh.formatters,
            namemap=mm.io.motchallenge_metric_names,
        )

        print(strsummary)

    @staticmethod
    def _get_gym_path():
        gym_dir = osp.dirname(__file__)
        while osp.basename(gym_dir) != 'mot_gym':
            if gym_dir == '/':
                raise Exception('Could not find mot_gym path')
            parent = osp.join(gym_dir, os.pardir)
            gym_dir = os.path.abspath(parent)
        return gym_dir

    @staticmethod
    def _visualize_box(img, text, box, color_index, emphasis=False):
        x0, y0, width, height = box 
        x0, y0, width, height = int(x0), int(y0), int(width), int(height)
        color = (_COLORS[color_index%80] * 255).astype(np.uint8).tolist()
        txt_color = (0, 0, 0) if np.mean(_COLORS[color_index%80]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        txt_size = cv2.getTextSize(text, font, 0.6, 1)[0]
        cv2.rectangle(img, (x0, y0), (x0+width, y0+height), color, 3 if emphasis else 1)

        txt_bk_color = (_COLORS[color_index%80] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.6, txt_color, thickness=1)
        return img