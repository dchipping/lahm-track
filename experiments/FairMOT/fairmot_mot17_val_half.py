from pathlib import Path
from fairmot_exp_utils import *

MODEL_PATH = '/home/dchipping/project/dan-track/mot-gallery-agent/motgym/trackers/FairMOT/models/fairmot_dla34.pth'
DATA_DIR = '/home/dchipping/project/dan-track/mot-gallery-agent/motgym/datasets/MOT17/val_half'
RESULTS_DIR = ''

conf_thres = 0.4
opt = opts().init(['mot', f'--load_model={MODEL_PATH}', f'--data_dir={DATA_DIR}',
                   f'--conf_thres={conf_thres}'])  # , '--buffer_size=100'])
seqs = sorted(filter(lambda x: not '.txt' in x, os.listdir(DATA_DIR)))

main(opt,
     data_root=DATA_DIR,
     seqs=seqs,
     exp_name=Path(__file__).stem,
     #  run_name='buffer_size_100',
     show_image=True,
     agent_path='greedy')
