import os
import os.path as osp
from math import ceil
import shutil

# This is NOT the conventional way to split the MOT datasets

curr_dir = osp.dirname(__file__)

train_dir = osp.join(curr_dir, 'train')
train_half_dir = osp.join(curr_dir, 'train_half')
val_half_dir = osp.join(curr_dir, 'val_half')

seqs = sorted(os.listdir(train_dir))
mid = ceil(len(seqs) / 2)

os.makedirs(osp.join(train_half_dir), exist_ok=True)
for seq in seqs[:mid]:
	src_path = osp.join(train_dir, seq)
	dst_path = osp.join(train_half_dir, seq)
	shutil.copytree(src_path, dst_path)

os.makedirs(osp.join(val_half_dir), exist_ok=True)
for seq in seqs[mid:]:
	src_path = osp.join(train_dir, seq)
	dst_path = osp.join(val_half_dir, seq)
	shutil.copytree(src_path, dst_path)
