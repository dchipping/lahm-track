# find . -name 'img1' | xargs -I % sh -c 'if [ $(ls % | wc -l) -lt 1800 ]; then echo % $(ls % | wc -l); fi'
import os
import shutil
from pathlib import Path

path = Path(__file__)
anno_path = os.path.join(path.parent, 'mot_annotations')
frames_path = os.path.join(path.parent, 'train', 'frames')
seqs = os.listdir(frames_path)

for seq in seqs:
    # Sequenence annotations
    seq_anno = os.path.join(anno_path, seq)
    dest_path = os.path.join(path.parent, 'train', seq)
    shutil.copytree(seq_anno, dest_path, dirs_exist_ok=True)

    # Move images
    imgs_path = os.path.join(frames_path, seq, 'rgb')
    imgs = os.listdir(imgs_path)

    dest_path = os.path.join(path.parent, 'train', seq, 'img1')
    os.makedirs(dest_path, exist_ok=True)
    for img in imgs:
        src_path = os.path.join(imgs_path, img)
        img_path = os.path.join(dest_path, img)
        shutil.move(src_path, img_path)

    os.removedirs(imgs_path)
    print('Moved: ' + seq)
