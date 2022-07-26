import csv
import os
import os.path as osp
import shutil

curr_dir = osp.dirname(__file__)

train_dir = osp.join(curr_dir, 'train')
train_half_dir = osp.join(curr_dir, 'train_half')
val_half_dir = osp.join(curr_dir, 'val_half')

seqs = os.listdir(train_dir)
for seq in seqs:
	img1_path = osp.join(train_dir, seq, 'img1')
	gt_path = osp.join(train_dir, seq, 'gt', 'gt.txt')
	frames = sorted(os.listdir(img1_path))
	gt_reader = csv.reader(open(gt_path, 'r'))

	seq_len = len(frames); mid = seq_len // 2
	train_half_frames = frames[:mid]
	val_half_frames = frames[mid:]

	os.makedirs(osp.join(train_half_dir, seq, 'img1'), exist_ok=True)
	for i, frame in enumerate(train_half_frames):
		src_path = osp.join(train_dir, seq, 'img1', frame)
		dst_path = osp.join(train_half_dir, seq, 'img1', '{:06d}.jpg'.format(i+1))
		shutil.copy(src_path, dst_path)

	os.makedirs(osp.join(val_half_dir, seq, 'img1'), exist_ok=True)
	for i, frame in enumerate(val_half_frames):
		src_path = osp.join(train_dir, seq, 'img1', frame)
		dst_path = osp.join(val_half_dir, seq, 'img1', '{:06d}.jpg'.format(i+1))
		shutil.copy(src_path, dst_path)

	os.makedirs(osp.join(train_half_dir, seq, 'gt'), exist_ok=True)
	os.makedirs(osp.join(val_half_dir, seq, 'gt'), exist_ok=True)
	train_half_gt = csv.writer(open(osp.join(train_half_dir, seq, 'gt', 'gt.txt'), 'w'))
	val_half_gt = csv.writer(open(osp.join(val_half_dir, seq, 'gt', 'gt.txt'), 'w'))
	
	tids = {}
	count = 1
	for row in gt_reader:
		frame_id = int(row[0])
		if frame_id <= mid:
			train_half_gt.writerow(row)
		else:
			if row[1] not in tids:
				tids[row[1]] = count
				count += 1
			row[0] = frame_id - mid
			row[1] = tids[row[1]]
			val_half_gt.writerow(row)

	# Update sequence lengths in info files
	info = open(osp.join(train_dir, seq, 'seqinfo.ini'), 'r').readlines()
	train_half_info = info[:]; train_half_info[4] = f'seqLength={len(train_half_frames)}\n'
	val_half_info = info[:]; val_half_info[4] = f'seqLength={len(val_half_frames)}\n'
	open(osp.join(train_half_dir, seq, 'seqinfo.ini'), 'w').write(''.join(train_half_info))
	open(osp.join(val_half_dir, seq, 'seqinfo.ini'), 'w').write(''.join(val_half_info))
