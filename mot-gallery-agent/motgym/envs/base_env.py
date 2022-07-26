from abc import abstractmethod
import gym
import random
import cv2
import os
import os.path as osp
import time
import numpy as np
import datetime as dt
import motmetrics as mm
from ._bbox_colors import _COLORS
from .utils.evaluation import Evaluator
from .utils.timer import Timer
from .utils.io import unzip_objs


class BasicMotEnv(gym.Env):
    def __init__(self, dataset, detections, seed=1):
        self.action_space = None
        self.observation_space = None
        
        random.seed(3)
        self.first_render = True
        self.timer = Timer()

        self.gym_path = BasicMotEnv._get_gym_path()
        self.data_dir = osp.join(self.gym_path, 'datasets', dataset)
        self.dets_dir = osp.join(self.gym_path, 'detections', detections)
        self.results_dir = BasicMotEnv._set_results_dir(self.gym_path)
        self.seqs = os.listdir(self.data_dir)

        self.tracker_args = None
        self.tracker = None

    def _reset_state(self):
        self.ep_reward = 0
        self.frame_id = 1
        self.online_targets = []
        self.track_idx = 0
        self.fps = None
        self.results = []

    def _reset_seq(self):
        self.seq = self.seqs[random.randint(0, len(self.seqs))]
        print(f'Loading data from: {osp.join(self.data_dir, self.seq)}')
        self._load_dataset(self.seq)
        self._load_detections(self.seq)

    def _load_dataset(self, seq):
        self.evaluator = Evaluator(self.data_dir, seq, 'mot')
        img1_path = osp.join(self.data_dir, seq, 'img1')
        self.images = sorted(map(lambda x: osp.join(img1_path, x), os.listdir(img1_path)))
        try:
            meta_info = open(osp.join(self.data_dir, seq, 'seqinfo.ini')).read()
            self.frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
            self.seq_len = int(meta_info[meta_info.find('seqLen') + 10:meta_info.find('\nimWidth')])
        except:
            print("Unable to load meta data")

    def _load_detections(self, seq):
        self.detections = np.load(osp.join(self.dets_dir, seq, 'dets.npz'))
        self.features = np.load(osp.join(self.dets_dir, seq, 'feats.npz'))

    @abstractmethod
    def reset(self):
        pass

    def _get_events(self, results):
        self.evaluator.reset_accumulator()

        frames = sorted(
            list(set(self.evaluator.gt_frame_dict.keys()) & set(results.keys())))
        for frame_id in frames:
            trk_objs = results.get(frame_id, [])
            trk_tlwhs, trk_ids = unzip_objs(trk_objs)[:2]
            self.evaluator.eval_frame(
                frame_id, trk_tlwhs, trk_ids, rtn_events=False)

        events = self.evaluator.acc.mot_events
        return events

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def render(self, mode="human"):
        pass

    def _init_rendering(self, shape):
        # Create empty frame on first load
        if self.first_render:
            black_img = np.zeros(shape, dtype=float)
            # scale = 1200/shape[1]
            # black_img = cv2.resize(black_img, None, fx = scale, fy = scale)
            cv2.imshow('env snapshot', black_img)
            cv2.waitKey(1)
            time.sleep(1)
            self.first_render = False

    def _display_frame(self, img, track_id):
        text = f'Frame {self.frame_id}, TrackID {track_id}, {self.fps} fps'
        # scale = 1200/img.shape[1]
        # img = cv2.resize(img, None, fx = scale, fy = scale)
        cv2.putText(img, text, (6, 22), cv2.FONT_HERSHEY_PLAIN,
                    1.25, (0, 0, 255), 2, cv2.LINE_8)
        cv2.imshow('env snapshot', img)
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()
        self.first_render = True

    @staticmethod
    def calc_fps(func):
        def inner(self, action):
            self.timer.tic()
            output = func(self, action)
            self.timer.toc()
            self.fps = round(1./self.timer.average_time, 2)
            return output
        return inner

    @staticmethod
    def _write_results(results, results_file, data_type):
        if data_type == 'mot':
            save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
        elif data_type == 'kitti':
            save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
        else:
            raise ValueError(data_type)

        with open(results_file, 'w') as f:
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
        print('save results to {}'.format(results_file))
        return results_file

    @staticmethod
    def _get_summary(eval, seq, results_file):
        eval.reset_accumulator()
        acc = eval.eval_file(results_file)
        metrics = mm.metrics.motchallenge_metrics
        mh = mm.metrics.create()
        summary = mh.compute(
            acc, 
            metrics=metrics,
            name=seq
        )

        strsummary = mm.io.render_summary(
            summary,
            formatters=mh.formatters,
            namemap=mm.io.motchallenge_metric_names,
        )

        print(strsummary)

    @staticmethod
    def _set_results_dir(root_path):
        timestamp = dt.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
        results_dir = osp.join(root_path, 'results', timestamp)
        os.makedirs(results_dir, exist_ok=True)
        return results_dir

    @staticmethod
    def _get_gym_path():
        gym_dir = osp.dirname(__file__)
        while osp.basename(gym_dir) != 'motgym':
            if gym_dir == '/':
                raise Exception('Could not find motgym path')
            parent = osp.join(gym_dir, os.pardir)
            gym_dir = os.path.abspath(parent)
        return gym_dir

    @staticmethod
    def _visualize_box(img, text, box, color_index, emphasis=False):
        x0, y0, width, height = box
        x0, y0, width, height = int(x0), int(y0), int(width), int(height)
        color = (_COLORS[color_index % 80] * 255).astype(np.uint8).tolist()
        txt_color = (0, 0, 0) if np.mean(
            _COLORS[color_index % 80]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        txt_size = cv2.getTextSize(text, font, 0.6, 1)[0]
        cv2.rectangle(img, (x0, y0), (x0+width, y0+height),
                      color, 3 if emphasis else 1)

        txt_bk_color = (_COLORS[color_index % 80] *
                        255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(
            img, text, (x0, y0 + txt_size[1]), font, 0.6, txt_color, thickness=1)
        return img
