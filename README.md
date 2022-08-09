# Reinforcement Learning Appearance History Management for Visual Multi-Object Tracking

- Tools for benchmarking and visualising tracking
- Includes mot-gallery-agent for training RL apperance management agent


## Setup Environment

```bash
git submodule update --init

conda create -n <tracker name e.g. FairMOT>

# Follow tracker's install guide e.g. see ./mot-gallery-agent/motgym/trackers/FairMOT/README.md

pip install -e mot-gallery-agent
```
Optional Move Dataset/Detections
```bash
cd mot-gallery-agent

mv ./motgym/datasets <target/dataset/path>
./tools/datasets_symbolic_link.sh <target/dataset/path>

mv ./motgym/detections <target/detections/path>
./tools/detections_symbolic_link.sh <target/detections/path>
```

## Setup Dataset and Detections
```bash
# Download datasets with instructions in [DATASETS.md](/motgym/datasets/DATASETS.md)
# Create features and detections using tracker in ./motgym/datasets/<tracker>/<gen_dets_script> e.g. gen_fairmot_jde.py

# Run a manual test after this using 
python ./tools/manual_test_sequential.py
```

## Acknowledgements

This code leans on the work of the authors of FairMOT (for baseline) and ByteTrack (for visulisation scripts).