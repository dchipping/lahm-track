import os
from pathlib import Path
from fairmot_exp_utils import *

AGENT_PATH = os.getcwd() + \
    '/mot-gallery-agent/results/fairmot_seq_ppo_mot20_train_half/2022-08-07T21-48-59/checkpoint'
MODEL_PATH = os.getcwd() + '/mot-gallery-agent/motgym/trackers/FairMOT/models/fairmot_dla34.pth'
DATA_DIR = os.getcwd() + '/mot-gallery-agent/motgym/datasets/MOT20/val_half'
RESULTS_DIR = ''

conf_thres = 0.4
opt = opts().init(['mot', f'--load_model={MODEL_PATH}', f'--data_dir={DATA_DIR}',
                   f'--conf_thres={conf_thres}'])
seqs = sorted(filter(lambda x: not '.txt' in x, os.listdir(DATA_DIR)))

main(opt,
     data_root=DATA_DIR,
     seqs=seqs,
     exp_name=Path(__file__).stem,
     # results_dir=RESULTS_DIR,
     show_image=False,
     agent_path=AGENT_PATH)
