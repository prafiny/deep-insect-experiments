#!/bin/bash

source set_tools/files.sh
get_set_info $2

if [ $# -gt 2 ]; then
    TRAIN_PARAM="${@:3}"
else
    TRAIN_PARAM=""
fi

cp "$SPLIT_LIB" set_tools/split.py
mkdir results
title=$(date +"%F_%H-%M-%S")_siftbow_${SET}
python train.py -t $IMG_FOLDER -c $TRAINING_CSV $TRAIN_PARAM > results/training.log 2>&1
python test.py -t $IMG_FOLDER -c $TESTING_CSV > results/testing.csv 2> results/testing.err
python report.py --truth-csv $WHOLE_CSV --result-csv testing.csv > results/report.html 2> results/report.err
