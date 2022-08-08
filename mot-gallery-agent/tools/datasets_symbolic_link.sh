# If detections are stored in a different location
# create a symbolic link to emulate a local folder
#
# Usage:
# cd ./path/to/mot-gallery-agent
# ./tools/detections_symbolic_link.sh [abs/path/to/detections]

DATA_DIR='./motgym/detections'

if [ -d $DATA_DIR ]; then
    echo "A 'detections' directory already exists!"
    exit
elif ! [ -d $1 ]; then
    echo "Invalid directory path"
    exit
fi

ln -s $1 ./motgym