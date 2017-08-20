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
. ~/dev/tensorflow_gpu/bin/activate
python train.py -t $IMG_FOLDER -c $TRAINING_CSV $TRAIN_PARAM > results/training.log 2>&1
python test.py -t $IMG_FOLDER -c $TESTING_CSV > results/testing.csv 2> results/testing.err
python report.py --truth-csv $WHOLE_CSV --result-csv testing.csv > results/report.html 2> results/report.err

FOLDER=~/experiments/${title}
mkdir -p $FOLDER
tar -zcvf $FOLDER/${title}.tar.gz .
cp -r results $FOLDER/results
cp results/report.html $FOLDER/report.html
mkdir $FOLDER/code
cp -r * $FOLDER/code/
rm . -r
