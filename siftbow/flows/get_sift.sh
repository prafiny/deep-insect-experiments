#!/bin/bash

source set_tools/files.sh
get_set_info $1

FOLDER=~/sets/$1/sift
mkdir per_img
python get_sift.py $IMG_FOLDER per_img > get_sift.log 2>&1

cp -r per_img $FOLDER
rm . -r
