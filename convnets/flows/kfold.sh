#!/bin/bash

source set_tools/files.sh
get_set_info $2

if [ $# -gt 2 ]; then
    TRAIN_PARAM="${@:3}"
else
    TRAIN_PARAM=""
fi

TEMPLATE="experiments/$1.py.m4"
SCRIPT="train.py"
m4 "$TEMPLATE" > "$SCRIPT"
cp "$MEANSTD" meanstd.pkl
cp "$SPLIT_LIB" set_tools/split.py

mkdir results
. ~/dev/tensorflow_gpu/bin/activate

python $SCRIPT --cross-validation --training $IMG_FOLDER --csvfile $WHOLE_CSV $TRAIN_PARAM > results/training.log 2>&1

title=$(date +"%F_%H-%M-%S")_${1}_${SET}_kfold
for combination in results/kfold/*; do
    python test.py -m $combination/model.h5 --image-folder $IMG_FOLDER -c $combination/testing.csv > $combination/results.csv 2> $combination/results.err
    python get_predictions.py -m $combination/model.h5 --image-folder $IMG_FOLDER -c $combination/testing.csv > $combination/dinet.csv 2> $combination/dinet.err
    sed '${/^$/d}' $combination/dinet.csv
    python report.py --truth-csv $WHOLE_CSV --result-csv $combination/results.csv --infos $combination/infos.pkl --name ${title} > $combination/report.html 2> $combination/report.err
done

python kfold_report.py --truth-csv $WHOLE_CSV --results-folder results/kfold/ --name ${title} > results/report.html 2> results/report.err

FOLDER=~/experiments/${title}
mkdir -p $FOLDER
tar -zcvf $FOLDER/${title}.tar.gz .
cp -r results $FOLDER/results
cp results/report.html $FOLDER/report.html
mkdir $FOLDER/code
cp -r * $FOLDER/code/
rm . -r
