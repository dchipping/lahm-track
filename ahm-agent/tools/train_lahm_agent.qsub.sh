# These are flags you must include - Two memory and one runtime.
# Runtime is either seconds or hours:min:sec

#$ -l tmem=2G
#$ -l h_vmem=16G
#$ -l h_rt=24:0:0 

#These are optional flags but you probably want them in all jobs

#$ -S /bin/bash
#$ -j y
#$ -N train_lahm_agent_mot17

# output directory for STDOUT file

#$ -o ~/run-log/

# Set resources CPU/GPU

#$ -pe smp 8
# #$ -l gpu=1

hostname
date

# See resources
echo GPUs: $(nvidia-smi -L)
echo Processors: $(nproc)

# conda env
export PATH=/home/$USER/miniconda3/bin:${PATH}
export LD_LIBRARY_PATH=/home/$USER/miniconda3/lib/:${LD_LIBRARY_PATH}
conda activate FairMOT
python --version

# Make scratch space
UNIQUEID=$(uuidgen)
UNIQUEID=${UNIQUEID:0:13}
BASEDIR="/scratch0/$USER/"
mkdir $BASEDIR
SCRATCH="${BASEDIR}${UNIQUEID}/"
mkdir $SCRATCH

# Directories for datasets and detections
DATADIR="${SCRATCH}datasets/"
mkdir $DATADIR
DETSDIR="${SCRATCH}detections/"
mkdir $DETSDIR
find $SCRATCH -maxdepth 2

# Directories for results
RESULTSDIR="${SCRATCH}results/"
mkdir $RESULTSDIR

# Download MOT17 data
mkdir "${DATADIR}MOT17/"
cd "${DATADIR}MOT17/"
wget -nv https://motchallenge.net/data/MOT17Det.zip
unzip -q MOT17Det.zip
rm MOT17Det.zip
rsync -ar --info=progress2 /home/$USER/repos/lahm-track/ahm-agent/tools/seperate_seqs.py .
python seperate_seqs.py

# Download MOT20 data
# mkdir "${DATADIR}MOT20/"
# cd "${DATADIR}MOT20/"
# wget https://motchallenge.net/data/MOT20.zip
# unzip MOT20.zip
# mv MOT20/train .
# mv MOT20/test .
# rm -d MOT20.zip MOT20
# rsync -ar --info=progress2 /home/chipping/repos/lahm-track/ahm-agent/tools/split_seqs.py .
# python split_seqs.py

# Download detections
cd $DETSDIR
gdown -q 1JlNKD1uFPXfs5mEYswsoa4AVfP0Hafyw # Google Drive Dets Download
unzip -q FairMOT.zip
rm FairMOT.zip

# Check file structure
echo "Data file structure:"
find $SCRATCH -maxdepth 3

# Run links
cd /home/$USER/repos/lahm-track/ahm-agent/
rm -d /home/$USER/repos/lahm-track/ahm-agent/motgym/datasets
rm -d /home/$USER/repos/lahm-track/ahm-agent/motgym/detections
./tools/datasets_symbolic_link.sh $(realpath $DATADIR)
./tools/detections_symbolic_link.sh $(realpath $DETSDIR)

# Run script
date
echo 'Starting training...'
python -u ./train/FairMOT/sequential/fairmot_seq_ppo_mot17_train_half.py $RESULTSDIR

# Move results to persistent store
echo 'Saving results...'
mkdir /home/$USER/train-results/
rsync -ar --info=progress2 $RESULTSDIR /home/$USER/train-results/$UNIQUEID

date
