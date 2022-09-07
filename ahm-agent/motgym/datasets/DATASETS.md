# Setup Instructions for Datasets

### [MOT17](https://motchallenge.net/data/MOT17Det)
```bash
cd MOT17
wget https://motchallenge.net/data/MOT17Det.zip
unzip MOT17Det.zip
rm MOT17Det.zip
python seperate_seqs.py
```

### [MOT20](https://motchallenge.net/data/MOT20)
```bash
cd MOT20
wget https://motchallenge.net/data/MOT20.zip
unzip MOT20.zip
mv MOT20/train .
mv MOT20/test .
rm -d MOT20.zip MOT20
python split_seqs.py
```

### [MOTSynth](https://motchallenge.net/data/MOTSynth-MOT-CVPR22/)
```bash
cd MOTSynth
wget https://motchallenge.net/data/MOTSynth_mot_annotations.zip
wget https://motchallenge.net/data/MOTSynth_1.zip https://motchallenge.net/data/MOTSynth_2.zip https://motchallenge.net/data/MOTSynth_3.zip
unzip MOTSynth_1.zip MOTSynth_2.zip MOTSynth_3.zip
rm MOTSynth_1.zip MOTSynth_2.zip MOTSynth_3.zip
mkdir train
python extract_frames.py --video_dir=./MOTSynth_1 --out_dir=./train
python extract_frames.py --video_dir=./MOTSynth_2 --out_dir=./train
python extract_frames.py --video_dir=./MOTSynth_3 --out_dir=./train
python reformat_mot17.py
```

### [DanceTrack](https://dancetrack.github.io/)
```bash
pip install gdown
mkdir dancetrack
cd dancetrack
gdown 1Jq6eH2n-WXMJLJ2AoKIZBtvdzKb2zkP5 12ybQtO1CRjyUctliWfysX34m66OX1Kvd 1AacPqd8iM2qD1OaYS9NklvKGyCl-vvZG
unzip train.zip val.zip test.zip
rm train.zip val.zip test.zip
```