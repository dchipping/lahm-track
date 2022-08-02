from pathlib import Path
from fairmot_exp_utils import *

AGENT_PATH = '/home/dchipping/ray_results/default/DQN_motgym:Mot17Env-v0_e3037_00000_0_2022-07-26_05-53-27/checkpoint_000035/checkpoint-35'
MODEL_PATH = '/home/dchipping/project/dan-track/mot-gallery-agent/motgym/trackers/FairMOT/models/fairmot_dla34.pth'
DATA_DIR = '/home/dchipping/project/dan-track/mot-gallery-agent/motgym/datasets/MOT17/val_half'

conf_thres = 0.4
opt = opts().init(['mot', f'--load_model={MODEL_PATH}', f'--data_dir={DATA_DIR}',
                    f'--conf_thres={conf_thres}'])

main(opt,
    data_root=DATA_DIR,
    seqs=sorted(os.listdir(DATA_DIR)),
    exp_name=Path(__file__).stem,
    show_image=False,
    agent_path=AGENT_PATH)
