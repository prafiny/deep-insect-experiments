#!/bin/bash

source set_tools/files.sh
get_set_info $1

cp "$SPLIT_LIB" set_tools/split.py
mkdir results
title=$(date +"%F_%H-%M-%S")_siftbow_${SET}_kfold
python train.py -t $SIFT_FOLDER -c $WHOLE_CSV > results/training.log 2>&1

for combination in results/kfold/*; do
    python test.py -m $combination/bof.pkl -t $SIFT_FOLDER -c $combination/testing.csv > $combination/results.csv 2> $combination/results.err
    #python get_predictions.py -m $combination/model.h5 --image-folder $IMG_FOLDER -c $combination/testing.csv > $combination/dinet.csv 2> $combination/dinet.err
    #sed '${/^$/d}' $combination/dinet.csv
    python report.py --truth-csv $WHOLE_CSV --result-csv $combination/results.csv > $combination/report.html 2> $combination/report.err
done

python kfold_report.py --truth-csv $WHOLE_CSV --results-folder results/kfold/ --name ${title} > results/report.html 2> results/report.err
