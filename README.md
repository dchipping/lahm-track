# Visual Multi-Object Tracking using Learned Appearance History Management (LAHM)
### A framework for training and evaluating Apperance History Management agents in a Multi-Object visual tracking context:
- `/experiments`: Evaluation scripts for running agents in the tracker on MOT Challenge datasets.
- `/mot-gallery-agent`: Package for training apperance management agents using variety of modern Reinforcement Learning methods.
- `/tools`: Results analysis tooling and visulisation scripts for qualatative anlysis.
- `/docs`: Research report and additional material webpage.

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

Optional FairMOT Model Download
```bash
pip install gdown
cd mot-gallery-agent/motgym/trackers/FairMOT
mkdir models
cd models
gdown 1iqRQjsG9BawIl8SlFomMg5iwkb6nqSpi
```

## Setup Dataset and Detections
### [Datasets](/motgym/datasets/DATASETS.md)
### [Detections](/mot-gallery-agent/motgym/detections/)

```bash
# Run a manual test after this using 
python ./tools/manual_test_sequential.py
```

## Acknowledgements

This code leans on the work of the authors of FairMOT (for baseline) and ByteTrack (for visulisation scripts).