import os
import cv2

def merge_visualization(agent_dir, baseline_dir, gt_dir, out_dir):
    os.makedirs(out_dir,  exist_ok=True)
    # if "mp4" in seq:
    #     continue
    os.makedirs(out_dir, exist_ok=True)
    frames = sorted(os.listdir(baseline_dir))
    for frame in frames:
        f_baseline_path = os.path.join(baseline_dir, frame)
        f_agent_path = os.path.join(agent_dir, frame)
        f_gt_path = os.path.join(gt_dir, frame)
        text1 = 'BASELINE'
        im1 = cv2.imread(f_baseline_path)
        cv2.putText(im1, text1, (245, 450), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2, cv2.LINE_AA)
        text2 = 'AGENT'
        im2 = cv2.imread(f_agent_path)
        cv2.putText(im2, text2, (245, 450), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2, cv2.LINE_AA)
        text3 = 'GT'
        im3 = cv2.imread(f_gt_path)
        cv2.putText(im3, text3, (300, 450), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2, cv2.LINE_AA)
        im_concat = cv2.hconcat([im2, im1, im3])
        f_out_dir = os.path.join(out_dir, frame)
        print(f_out_dir)
        cv2.imwrite(f_out_dir, im_concat)
        
    cmd = "ffmpeg -framerate 14 -pattern_type glob -i '{}/*.jpg' -c:v libx264 -pix_fmt yuv420p {}/merged.mp4".format(out_dir, out_dir)
    os.popen(cmd)

out_dir = '/home/dchipping/project/dan-track/videos'
agent_dir = '/home/dchipping/project/dan-track/mot-gym/mot_gym/data/MOT17/outputs/agent/MOT17-05'
baseline_dir = '/home/dchipping/project/dan-track/mot-gym/mot_gym/data/MOT17/outputs/baseline/MOT17-05'
gt_dir = '/home/dchipping/project/dan-track/mot-gym/mot_gym/data/MOT17/outputs/gt-imgs/MOT17-05'
merge_visualization(agent_dir, baseline_dir, gt_dir, out_dir)
