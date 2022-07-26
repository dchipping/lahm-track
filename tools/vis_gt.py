import csv
import os
import os.path as osp
from collections import defaultdict
import cv2
import mot_gym
from FairMOT.src.lib.tracking_utils.visualization import plot_tracking

# gt_path = '/home/dchipping/project/dan-track/mot-gym/mot_gym/data/MOT17/train/MOT17-05/gt/gt.txt'
# gt_path = '/home/dchipping/project/dan-track/results/agent-2022-07-22T07-19-48/MOT17-05.txt'
gt_path = '/home/dchipping/project/dan-track/results/baseline-2022-07-21T23-20-29/MOT17-05.txt'
data_dir = '/home/dchipping/project/dan-track/mot-gym/mot_gym/data/MOT17/train/MOT17-05'
# output_dir = '/home/dchipping/project/dan-track/mot-gym/mot_gym/data/MOT17/outputs/gt-imgs'
output_dir = '/home/dchipping/project/dan-track/mot-gym/mot_gym/data/MOT17/outputs/baseline'
if not osp.exists(output_dir):
    os.makedirs(output_dir)

img_dir = osp.join(data_dir, 'img1')
img_files = sorted(os.listdir(img_dir))

gt_tids = defaultdict(list)
gt_tlwh = defaultdict(list)
with open(gt_path) as f:
	reader = csv.reader(f)
	for row in reader:
		l = float(row[2])
		t = float(row[3])
		w = float(row[4])
		h = float(row[5])
		gt_tids[int(row[0])].append(int(row[1]))
		gt_tlwh[int(row[0])].append((l, t, w, h))

for i, file in enumerate(img_files):
    frame_id = i + 1
    img_path = osp.join(img_dir, file)
    img0 = cv2.imread(img_path)
    tids = gt_tids[frame_id]
    tlwhs = gt_tlwh[frame_id]
    online_im = plot_tracking(img0, tlwhs, tids, frame_id=frame_id, fps=0.)
    save = os.path.join(output_dir, '{:05d}.jpg'.format(frame_id))
    print(save)
    cv2.imwrite(save, online_im)