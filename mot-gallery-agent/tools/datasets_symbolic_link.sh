# If datasets are stored in a different location
# create a symbolic link to emulate a local folder
#
# Usage:
# cd ./path/to/mot-gallery-agent
# ./tools/datasets_symbolic_link.sh [abs/path/to/datasets]

DATA_DIR='./motgym/datasets'

if ! [ -d $DATA_DIR ]; then
    echo "A 'datasets' directory already exists!"
    exit
elif ! [ -d $1 ]; then
    echo "Invalid directory path"
    exit
fi

ln -s $1 ./motgym