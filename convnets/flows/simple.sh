#!/bin/bash

source set_tools/files.sh
get_set_info $2

TEMPLATE="experiments/$1.py.m4"
SCRIPT="train.py"
m4 "$TEMPLATE" > "$SCRIPT"
cp "$MEANSTD" meanstd.pkl
cp "$SPLIT_LIB" set_tools/split.py
mkdir results
. ~/dev/tensorflow_gpu/bin/activate
title=$(date +"%F_%H-%M-%S")_${1}_${SET}
python $SCRIPT --training $IMG_FOLDER --csvfile $TRAINING_CSV > results/training.log 2>&1
python test.py --image-folder $IMG_FOLDER -c $TESTING_CSV > results/results.csv 2> results/results.err

python get_predictions.py --image-folder $IMG_FOLDER -c $REDUCED_CSV > results/dinet.csv 2> results/dinet.err
sed '${/^$/d}' results/dinet.csv

python report.py --truth-csv $WHOLE_CSV --result-csv results/results.csv --infos infos.pkl --name ${title} > results/report.html 2> results/report.err

FOLDER=~/experiments/${title}
tar -zcvf ${title}.tar.gz .
mkdir -p $FOLDER
mv ${title}.tar.gz $FOLDER/
mv results $FOLDER/results
mkdir $FOLDER/code
cp -r * $FOLDER/code/
cp $FOLDER/results/report.html $FOLDER/
cp {model.txt,model.png} $FOLDER/
rm . -r
