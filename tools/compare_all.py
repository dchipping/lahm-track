import os
import cv2
import os.path as osp

def merge_visualization(agent_dir, baseline_dir, rdm_dir, gt_dir, out_dir, filename):
    # Check if video exists at output
    if os.path.exists(os.path.join(out_dir, f'{filename}.mp4')):
        raise FileExistsError(f'Output video file {filename}.mp4 already exists')

    # Create temporary folder for merged images
    tmp_dir = os.path.join(out_dir, 'tmp')
    if os.path.exists(tmp_dir):
        files = os.listdir(tmp_dir)
        for file in files:
            os.remove(os.path.join(tmp_dir, file))
    else:
        os.makedirs(tmp_dir, exist_ok=True)

    # Merge images together
    frames = sorted(os.listdir(baseline_dir))
    for frame in frames:
        f_baseline_path = os.path.join(baseline_dir, frame)
        f_agent_path = os.path.join(agent_dir, frame)
        f_gt_path = os.path.join(gt_dir, frame)
        f_rdm_path = os.path.join(rdm_dir, frame)
        text1 = 'BASELINE'
        im1 = cv2.imread(f_baseline_path)
        cv2.putText(im1, text1, (8, 40), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 3, cv2.LINE_8)
        text2 = 'AGENT'
        im2 = cv2.imread(f_agent_path)
        cv2.putText(im2, text2, (8, 42), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 3, cv2.LINE_8)
        iml_concat = cv2.hconcat([im1, im2])
        text3 = 'RANDOM'
        im3 = cv2.imread(f_rdm_path)
        cv2.putText(im3, text3, (8, 40), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 3, cv2.LINE_8)
        text4 = 'GT'
        im4 = cv2.imread(f_gt_path)
        cv2.putText(im4, text4, (8, 40), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 3, cv2.LINE_8)
        imr_concat = cv2.hconcat([im3, im4])
        im_concat = cv2.vconcat([iml_concat, imr_concat])
        f_out_dir = os.path.join(tmp_dir, frame)
        print(f_out_dir)
        cv2.imwrite(f_out_dir, im_concat)

    # Ensure ffmpeg is installed 
    cmd = "ffmpeg -framerate 5 -pattern_type glob -i '{}/*.jpg' -c:v libx264 -pix_fmt yuv420p {}/{}.mp4".format(tmp_dir, out_dir, filename)
    os.popen(cmd)

if __name__ == "__main__":
    filename = 'all-mot17-val-ppo'
    out_dir = '/home/dchipping/project/dan-track/videos'

    agent_dir = '/home/dchipping/project/dan-track/videos/images/jde_agent_mot17_val_half/ppo_mot17_val_half'
    baseline_dir = '/home/dchipping/project/dan-track/videos/images/jde_mot17_val_half/2022-09-10T12-58-07'
    rdm_dir = '/home/dchipping/project/dan-track/videos/images/jde_random_mot17_val_half/2022-09-10T14-39-47'
    gt_dir = '/home/dchipping/project/dan-track/videos/gt_images/MOT17/val_half'

    # merge_visualization(agent_dir, baseline_dir, rdm_dir, gt_dir, out_dir, filename)
    seqs = sorted(os.listdir(agent_dir))
    for seq in seqs:
        merge_visualization(
            osp.join(agent_dir, seq),
            osp.join(baseline_dir, seq),
            osp.join(rdm_dir, seq),
            osp.join(gt_dir, seq),
            out_dir,
            f'{filename}-{seq}')

