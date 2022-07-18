#!/bin/bash
# Usage: ./tools/trackid_freq.sh [trackid] [path/to/results.txt] 
# e.g. ./tools/trackid_freq.sh 20 ./data/MOT17/train/MOT17-05/results/results.txt 

IFS=,
count=1

while read -r frame track_id extra
do
    if [[ $track_id = $1 ]]; then
        count=$((count+1))
    fi
done < $2

echo "TrackID" $1 "seen in" $count "frames"
