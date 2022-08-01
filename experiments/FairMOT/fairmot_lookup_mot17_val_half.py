import datetime as dt
from pathlib import Path
from exp_utils import *

EXP_NAME = ''
MODEL_PATH = '/home/dchipping/project/dan-track/mot-gallery-agent/motgym/trackers/FairMOT/models/fairmot_dla34.pth'
DATA_DIR = '/home/dchipping/project/dan-track/mot-gallery-agent/motgym/datasets/MOT17/val_half'

conf_thres = 0.4
opt = opts().init(['mot', f'--load_model={MODEL_PATH}', f'--data_dir={DATA_DIR}',
                    f'--conf_thres={conf_thres}'])
exp_name = osp.join(Path(__file__).stem, 
    EXP_NAME if EXP_NAME else f'{dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")}')

main(opt,
    data_root=DATA_DIR,
    seqs=sorted(os.listdir(DATA_DIR)),
    exp_name=exp_name,
    show_image=False,
    lookup_gallery=True,
    agent_path='greedy')
