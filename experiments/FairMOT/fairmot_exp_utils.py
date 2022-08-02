import datetime as dt
import logging
import os
import os.path as osp
from pathlib import Path

import motmetrics as mm
import numpy as np
import torch
import cv2
import sys

import motgym.trackers.FairMOT.src._init_paths
import datasets.dataset.jde as datasets
from modified.fairmot_agent import AgentJDETracker
from tracking_utils import visualization as vis
from opts import opts
from tracking_utils.evaluation import Evaluator
from tracking_utils.log import logger
from tracking_utils.timer import Timer
from tracking_utils.utils import mkdir_if_missing

# Update sys path with tools dir
path = os.path.join(os.getcwd(), 'tools')
if path not in sys.path:
    sys.path.insert(0, path)
from TrackEval import trackeval


def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(
                    frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def eval_seq(opt, dataloader, data_type, result_filename, show_image=True,
             frame_rate=30, use_cuda=True, lookup_gallery=False, agent_path=None):
    tracker = AgentJDETracker(opt, frame_rate=frame_rate,
                              lookup_gallery=lookup_gallery, agent_path=agent_path)
    timer = Timer()
    results = []
    frame_id = 0

    for i, (path, img, img0) in enumerate(dataloader):
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(
                frame_id, 1. / max(1e-5, timer.average_time)))

        # run tracking
        timer.tic()
        if use_cuda:
            blob = torch.from_numpy(img).cuda().unsqueeze(0)
        else:
            blob = torch.from_numpy(img).unsqueeze(0)
        online_targets = tracker.update(blob, img0)
        online_tlwhs = []
        online_ids = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
        timer.toc()
        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))
        if show_image:
            online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
                                          fps=1. / timer.average_time)
            cv2.imshow('online_im', online_im)
            cv2.waitKey(1)
        frame_id += 1

    # save results
    write_results(result_filename, results, data_type)
    return frame_id, timer.average_time, timer.calls


def main(opt, data_root='/data/MOT16/train', seqs=('MOT16-05',), exp_name='demo',
         show_image=True, lookup_gallery=False, agent_path=None, run_name='original'):
    logger.setLevel(logging.INFO)
    run_name = run_name if run_name else dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    result_root = os.path.join(os.getcwd(), 'results', exp_name, run_name)
    mkdir_if_missing(result_root)
    data_type = 'mot'

    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []
    # for seq in seqs:
    #     logger.info('start seq: {}'.format(seq))
    #     dataloader = datasets.LoadImages(
    #         osp.join(data_root, seq, 'img1'), opt.img_size)
    #     result_filename = os.path.join(result_root, '{}.txt'.format(seq))
    #     meta_info = open(os.path.join(data_root, seq, 'seqinfo.ini')).read()
    #     frame_rate = int(meta_info[meta_info.find(
    #         'frameRate') + 10:meta_info.find('\nseqLength')])
    #     nf, ta, tc = eval_seq(opt, dataloader, data_type, result_filename,
    #                           show_image=show_image, frame_rate=frame_rate, lookup_gallery=lookup_gallery, agent_path=agent_path)
    #     n_frame += nf
    #     timer_avgs.append(ta)
    #     timer_calls.append(tc)

    #     # eval
    #     logger.info('Evaluate seq: {}'.format(seq))
    #     evaluator = Evaluator(data_root, seq, data_type)
    #     accs.append(evaluator.eval_file(result_filename))

    # timer_avgs = np.asarray(timer_avgs)
    # timer_calls = np.asarray(timer_calls)
    # all_time = np.dot(timer_avgs, timer_calls)
    # avg_time = all_time / np.sum(timer_calls)
    # logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(
    #     all_time, 1.0 / avg_time))

    # # get summary
    # metrics = mm.metrics.motchallenge_metrics
    # mh = mm.metrics.create()
    # summary = Evaluator.get_summary(accs, seqs, metrics)
    # strsummary = mm.io.render_summary(
    #     summary,
    #     formatters=mh.formatters,
    #     namemap=mm.io.motchallenge_metric_names
    # )
    # print(strsummary)
    # Evaluator.save_summary(summary, os.path.join(
    #     result_root, 'summary_{}.xlsx'.format(run_name)))

    seqmap_path = osp.join(data_root, 'seqmap.txt')
    if not osp.exists(seqmap_path):
        seqs = map(lambda x: x + '\n', os.listdir(data_root))
        with open(seqmap_path, 'w') as seqmap_file:
            seqmap_file.write('name\n')
            seqmap_file.writelines(seqs)

    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config['DISPLAY_LESS_PROGRESS'] = False
    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config['GT_FOLDER'] = data_root
    dataset_config['TRACKERS_FOLDER'] = osp.join(os.getcwd(), 'results')
    dataset_config['TRACKERS_TO_EVAL'] = [osp.join(exp_name, run_name)]
    dataset_config['TRACKER_SUB_FOLDER'] = ''
    dataset_config['SPLIT_TO_EVAL'] = Path(data_root).name
    dataset_config['OUTPUT_FOLDER'] = osp.join(os.getcwd(), 'results')
    dataset_config['SKIP_SPLIT_FOL'] = True
    dataset_config['SEQMAP_FILE'] = osp.join(data_root, 'seqmap.txt')

    # Run code
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = [trackeval.metrics.HOTA()]
    evaluator.evaluate(dataset_list, metrics_list)
