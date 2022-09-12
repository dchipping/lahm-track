import csv
import os
import os.path as osp
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)


def visualize_box(img, text, box, color_index):
    x0, y0, width, height = box
    x0, y0, width, height = int(x0), int(y0), int(width), int(height)
    color = (_COLORS[color_index % 80] * 255).astype(np.uint8).tolist()
    txt_color = (0, 0, 0) if np.mean(
        _COLORS[color_index % 80]) > 0.5 else (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    txt_size = cv2.getTextSize(text, font, 0.6, 1)[0]
    cv2.rectangle(img, (x0, y0), (x0+width, y0+height),
                  color, 2)

    txt_bk_color = (_COLORS[color_index % 80] *
                    255 * 0.7).astype(np.uint8).tolist()
    cv2.rectangle(
        img,
        (x0, y0 + 1),
        (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
        txt_bk_color,
        -1
    )
    cv2.putText(
        img, text, (x0, y0 + txt_size[1]), font, 0.6, txt_color, thickness=1)
    return img


if __name__ == "__main__":
    data_dir = '/home/dchipping/project/dan-track/ahm-agent/motgym/datasets/MOT20/val_half'
    output_dir = '/home/dchipping/project/dan-track/videos/gt_images'

    seqs = sorted(os.listdir(data_dir))
    for seq in seqs:
        p = Path(data_dir)
        img_output_dir = osp.join(output_dir, p.parent.name, p.name, seq)
        os.makedirs(img_output_dir, exist_ok=True)

        img_dir = osp.join(data_dir, seq, 'img1')
        img_files = sorted(os.listdir(img_dir))

        tids = defaultdict(list)
        tlwhs = defaultdict(list)
        results_path = osp.join(data_dir, seq, 'gt', 'gt.txt')
        with open(results_path) as f:
            reader = csv.reader(f)
            for row in reader:
                l = float(row[2])
                t = float(row[3])
                w = float(row[4])
                h = float(row[5])
                tids[int(row[0])].append(int(row[1]))
                tlwhs[int(row[0])].append((l, t, w, h))

        for i, file in enumerate(img_files):
            frame_id = i + 1
            img_path = osp.join(img_dir, file)
            img0 = cv2.imread(img_path)
            curr_tids = tids[frame_id]
            curr_tlwhs = tlwhs[frame_id]
            for tid, tlwh in zip(curr_tids, curr_tlwhs):
                img0 = visualize_box(img0, str(tid), tlwh, tid)
            img_name = osp.join(img_output_dir, '{:05d}.jpg'.format(frame_id))
            print(img_name)
            cv2.imwrite(img_name, img0)
