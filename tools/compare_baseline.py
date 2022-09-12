import os
import cv2
import os.path as osp

def merge_visualization(agent_dir, baseline_dir, out_dir, filename):
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
        text1 = 'BASELINE'
        im1 = cv2.imread(f_baseline_path)
        cv2.putText(im1, text1, (8, 40), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 3, cv2.LINE_8)
        text2 = 'AGENT'
        im2 = cv2.imread(f_agent_path)
        cv2.putText(im2, text2, (8, 42), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 3, cv2.LINE_8)
        im_concat = cv2.hconcat([im2, im1])
        f_out_dir = os.path.join(tmp_dir, frame)
        print(f_out_dir)
        cv2.imwrite(f_out_dir, im_concat)

    # Ensure ffmpeg is installed 
    cmd = "ffmpeg -framerate 5 -pattern_type glob -i '{}/*.jpg' -c:v libx264 -pix_fmt yuv420p {}/{}.mp4".format(tmp_dir, out_dir, filename)
    os.popen(cmd)

if __name__ == "__main__":
    out_dir = '/home/dchipping/project/dan-track/videos'

    filename = 'agent-vs-baseline'
    agent_dir = '/home/dchipping/project/dan-track/videos/images/jde_agent_mot17_test/ppo_mot17_val_half'
    baseline_dir = '/home/dchipping/project/dan-track/videos/images/jde_mot17_test/2022-09-11T03-46-14'

    # merge_visualization(agent_dir, baseline_dir, out_dir, filename)
    seqs = sorted(os.listdir(agent_dir))
    for seq in seqs:
        merge_visualization(
            osp.join(agent_dir, seq),
            osp.join(baseline_dir, seq),
            out_dir,
            f'{filename}-{seq}')
