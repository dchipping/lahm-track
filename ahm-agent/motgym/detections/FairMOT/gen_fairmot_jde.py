import numpy as np
import torch
import torch.nn.functional as F
import os
import os.path as osp
from pathlib import Path

import motgym
import FairMOT.src._init_paths
import datasets.dataset.jde as datasets
from opts import opts
from models import *
from models.decode import mot_decode
from models.utils import _tranpose_and_gather_feat
from tracking_utils.utils import *
from tracker.multitracker import JDETracker


class CachedJDETracker(JDETracker):
    def __init__(self, opt, frame_rate):
        super().__init__(opt, frame_rate)

    def jde_only(self, im_blob, img0):
        self.frame_id += 1
        if self.frame_id % 50 == 0 or self.frame_id == 1:
            print(f"Processing frame {self.frame_id}")

        width = img0.shape[1]
        height = img0.shape[0]
        inp_height = im_blob.shape[2]
        inp_width = im_blob.shape[3]
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
        meta = {'c': c, 's': s,
                'out_height': inp_height // self.opt.down_ratio,
                'out_width': inp_width // self.opt.down_ratio}

        ''' Step 1: Network forward, get detections & embeddings'''
        with torch.no_grad():
            output = self.model(im_blob)[-1]
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            id_feature = output['id']
            id_feature = F.normalize(id_feature, dim=1)

            reg = output['reg'] if self.opt.reg_offset else None
            dets, inds = mot_decode(
                hm, wh, reg=reg, ltrb=self.opt.ltrb, K=self.opt.K)
            id_feature = _tranpose_and_gather_feat(id_feature, inds)
            id_feature = id_feature.squeeze(0)
            id_feature = id_feature.cpu().numpy()

        dets = self.post_process(dets, meta)
        dets = self.merge_outputs([dets])[1]

        remain_inds = dets[:, 4] > self.opt.conf_thres
        dets = dets[remain_inds]
        id_feature = id_feature[remain_inds]

        return dets, id_feature


if __name__ == "__main__":
    conf_thres = 0.4  # FairMOT authors achieve SOTA on MOT17/20 with 0.4
    model_path = '/home/dchipping/project/dan-track/mot-gallery-agent/motgym/trackers/FairMOT/models/fairmot_dla34.pth'
    data_dir = '/home/dchipping/project/dan-track/mot-gallery-agent/motgym/datasets/MOTSynth/train'

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

        tracker_args = opts().init(
            ['mot', f'--load_model={model_path}', f'--conf_thres={conf_thres}'])
        tracker = CachedJDETracker(tracker_args, frame_rate)

        dets = {}
        feats = {}
        print(f"Starting the processing {seq} frames...")
        for i, (path, img, img0) in enumerate(dataloader):
            try:
                blob = torch.from_numpy(img).cuda().unsqueeze(0)
            except:
                blob = torch.from_numpy(img).unsqueeze(0)
            det, id_feature = tracker.jde_only(blob, img0)

            frame_id = i+1
            dets[str(frame_id)] = det
            feats[str(frame_id)] = id_feature

        print(f"Saving {seq} JDE to: {output_dir}")
        dets_file = osp.join(output_dir, 'dets.npz')
        feats_file = osp.join(output_dir, 'feats.npz')
        np.savez(dets_file, **dets)
        np.savez(feats_file, **feats)
