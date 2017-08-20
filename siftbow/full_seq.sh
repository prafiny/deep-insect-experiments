#!/bin/bash

python train.py -t /home/maxime/set/norecalib_scaled -c /home/maxime/set/CSVs/balanced_shotwise_training.csv > training.log 2>&1
python test.py -t /home/maxime/set/norecalib_scaled/ -c /home/maxime/set/CSVs/balanced_shotwise_testing.csv > testing.csv 2> testing.err
