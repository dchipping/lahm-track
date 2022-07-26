#!/bin/bash
# List all track_ids present in a given frame
# Usage: ./tools/frame_trackids.sh [frame_id] [path/to/results.txt] 
# e.g. ./tools/frame_trackids.sh 12 ./data/MOT17/train/MOT17-05/results/results.txt 

IFS=,
echo "Frame" $1 "TrackIDs:"

while read -r frame track_id extra
do
    if [[ $frame = $1 ]]; then
        echo $track_id
    fi
done < $2
