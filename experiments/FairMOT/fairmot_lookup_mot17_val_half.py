from pathlib import Path
from fairmot_exp_utils import *

MODEL_PATH = '/home/dchipping/project/dan-track/ahm-agent/motgym/trackers/FairMOT/models/fairmot_dla34.pth'
DATA_DIR = '/home/dchipping/project/dan-track/ahm-agent/motgym/datasets/MOT17/val_half'
RESULTS_DIR = ''

conf_thres = 0.4
opt = opts().init(['mot', f'--load_model={MODEL_PATH}', f'--data_dir={DATA_DIR}',
                   f'--conf_thres={conf_thres}'])
seqs = sorted(filter(lambda x: not '.txt' in x, os.listdir(DATA_DIR)))

main(opt,
     data_root=DATA_DIR,
     seqs=seqs,
     exp_name=Path(__file__).stem,
     # run_name='bug_fix',
     # results_dir=RESULTS_DIR,
     show_image=False,
     lookup_gallery=10,
     agent_path='greedy')
