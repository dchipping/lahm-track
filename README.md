# Multi-Object Tracking using Learned Appearance History Management (LAHM)
### A reinforcement learning framework for training and evaluating Apperance History Management agents in a multi-object tracking context:
- `/ahm-agent`: Package for training apperance history management (AHM) agents using variety of modern Reinforcement Learning methods.
- `/docs`: Research report and additional material webpage.
- `/experiments`: Evaluation scripts for running agents in the tracker on MOT Challenge datasets.
- `/tools`: Result analysis tooling and visulisation scripts for qualatative analysis.

## Setup Environment

```bash
git submodule update --init

conda create -n <tracker name e.g. FairMOT>

# Follow tracker's install guide e.g. see ./ahm-agent/motgym/trackers/FairMOT/README.md

pip install -e ahm-agent
```
Optional Move Dataset/Detections
```bash
cd ahm-agent

mv ./motgym/datasets <target/dataset/path>
./tools/datasets_symbolic_link.sh <target/dataset/path>

mv ./motgym/detections <target/detections/path>
./tools/detections_symbolic_link.sh <target/detections/path>
```

Optional FairMOT Model Download
```bash
pip install gdown
cd ahm-agent/motgym/trackers/FairMOT
mkdir models
cd models
gdown 1iqRQjsG9BawIl8SlFomMg5iwkb6nqSpi
```

## Setup Dataset and Detections
### [Datasets](/ahm-agent/motgym/datasets/DATASETS.md)
### [Detections](/ahm-agent/motgym/detections/DETECTIONS.md)

```bash
# Run a manual test after this using 
python ./tools/manual_test_sequential.py
```

## Acknowledgements

This code leans on the work of the authors of FairMOT (for baseline) and ByteTrack (for visulisation scripts).
