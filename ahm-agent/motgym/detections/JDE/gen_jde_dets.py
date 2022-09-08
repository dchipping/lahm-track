import argparse
import os
import os.path as osp
import sys
from pathlib import Path

import motgym
import torch
import numpy as np

JDE = __import__('Towards-Realtime-MOT')
sys.path.insert(0, JDE.__path__[0])

from tracker.multitracker import JDETracker
from utils import datasets
from utils.utils import *
from utils.parse_config import parse_model_cfg


class CachedJDETracker(JDETracker):
    def __init__(self, opt, frame_rate):
        super().__init__(opt, frame_rate)

    def jde_only(self, im_blob, img0):
        self.frame_id += 1
        if self.frame_id % 10 == 0 or self.frame_id == 1:
            print(f"Processing frame {self.frame_id}")

        # ''' Step 1: Network forward, get detections & embeddings'''
        with torch.no_grad():
            pred = self.model(im_blob)
        # pred is tensor of all the proposals (default number of proposals: 54264). Proposals have information associated with the bounding box and embeddings
        pred = pred[pred[:, :, 4] > self.opt.conf_thres]
        
        if len(pred) > 0:
            dets = non_max_suppression(pred.unsqueeze(0), self.opt.conf_thres, self.opt.nms_thres)[0].cpu()
            # Final proposals are obtained in dets. Information of bounding box and embeddings also included
            # Next step changes the detection scales
            scale_coords(self.opt.img_size, dets[:, :4], img0.shape).round()

        return dets

if __name__ == "__main__":
    conf_thres = 0.4  # FairMOT authors achieve SOTA on MOT17/20 with 0.4
    config_path = '/home/dchipping/project/dan-track/ahm-agent/motgym/trackers/Towards-Realtime-MOT/cfg/yolov3_1088x608.cfg' # 1088x608 864x480
    model_path = '/home/dchipping/project/dan-track/ahm-agent/motgym/trackers/Towards-Realtime-MOT/models/jde.1088x608.uncertainty.pt'
    data_dir = '/home/dchipping/project/dan-track/ahm-agent/motgym/datasets/MOT20/train_half'

    parser = argparse.ArgumentParser(prog='gen_jde_dets.py')
    parser.add_argument('--cfg', type=str, default=config_path, help='cfg file path')
    parser.add_argument('--weights', type=str, default=model_path, help='path to weights file')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
    parser.add_argument('--conf-thres', type=float, default=conf_thres, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.4, help='iou threshold for non-maximum suppression')
    parser.add_argument('--min-box-area', type=float, default=200, help='filter out tiny boxes')
    parser.add_argument('--track-buffer', type=int, default=30, help='tracking buffer')
    opts = parser.parse_args()

    cfg_dict = parse_model_cfg(opts.cfg)
    opts.img_size = [int(cfg_dict[0]['width']), int(cfg_dict[0]['height'])]

    seqs = sorted(os.listdir(data_dir))
    for seq in seqs:
        p = Path(data_dir)
        output_dir = osp.join(osp.dirname(__file__),
                              p.parent.name, p.name, seq)
        if osp.exists(output_dir):
            print('Skipping %s' % seq)
            continue
        else:
            os.makedirs(output_dir)

        img1_path = osp.join(data_dir, seq, 'img1')
        dataloader = datasets.LoadImages(img1_path)
        meta_info = open(osp.join(data_dir, seq, 'seqinfo.ini')).read()
        frame_rate = int(meta_info[meta_info.find(
            'frameRate') + 10:meta_info.find('\nseqLength')])

        tracker_args = opts
        tracker = CachedJDETracker(tracker_args, frame_rate)

        dets = {}
        print(f"Starting the processing {seq} frames...")
        for i, (path, img, img0) in enumerate(dataloader):
            try:
                blob = torch.from_numpy(img).cuda().unsqueeze(0)
            except:
                blob = torch.from_numpy(img).unsqueeze(0)
            det = tracker.jde_only(blob, img0)

            frame_id = i+1
            dets[str(frame_id)] = det

        print(f"Saving {seq} dets to: {output_dir}")
        dets_file = osp.join(output_dir, 'dets.npz') # 0:5 is bbox, 6: is embedding
        np.savez(dets_file, **dets)
