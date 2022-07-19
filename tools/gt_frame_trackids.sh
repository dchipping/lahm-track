#!/bin/bash
# Usage: ./tools/trackid_freq.sh [trackid] [path/to/results.txt] 
# e.g. ./tools/trackid_freq.sh 20 ./data/MOT17/train/MOT17-05/results/results.txt 

IFS=,
echo "Frame" $1 "TrackIDs:"

while read -r frame track_id extra
do
    if [[ $frame = $1 ]]; then
        echo $track_id
    fi
done < $2
