import argparse
from pathlib import Path
from jde_exp_utils import *

CONFIG_PATH = '/home/dchipping/project/dan-track/ahm-agent/motgym/trackers/Towards-Realtime-MOT/cfg/yolov3_1088x608.cfg'
MODEL_PATH = '/home/dchipping/project/dan-track/ahm-agent/motgym/trackers/Towards-Realtime-MOT/models/jde.1088x608.uncertainty.pt'
DATA_DIR = '/home/dchipping/project/dan-track/ahm-agent/motgym/datasets/MOT17/val_half'
RESULTS_DIR = ''

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, default=CONFIG_PATH, help='cfg file path')
parser.add_argument('--weights', type=str, default=MODEL_PATH, help='path to weights file')
parser.add_argument('--iou-thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
parser.add_argument('--nms-thres', type=float, default=0.4, help='iou threshold for non-maximum suppression')
parser.add_argument('--min-box-area', type=float, default=200, help='filter out tiny boxes')
parser.add_argument('--track-buffer', type=int, default=30, help='tracking buffer')

seqs = sorted(filter(lambda x: not '.txt' in x, os.listdir(DATA_DIR)))

main(parser.parse_args(),
     data_root=DATA_DIR,
     seqs=seqs,
     exp_name=Path(__file__).stem,
     #  run_name='buffer_size_100',
     show_image=True,
     agent_path='greedy')
