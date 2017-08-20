import argparse as ap
import cv2
import imutils 
import numpy as np
import os
import csv
import sys
from sklearn.externals import joblib

sift_ext = cv2.xfeatures2d.SIFT_create()
def get_sift(image_path):
    im = cv2.imread(image_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    kps, des = sift_ext.detectAndCompute(im, None)
    del im
    return des

# Get the path of the training set
parser = ap.ArgumentParser()
parser.add_argument("folder")
parser.add_argument("out")
args = vars(parser.parse_args())

for fil in os.listdir(args["folder"]):
    path = os.path.join(args["folder"], fil)
    des = get_sift(path)
    joblib.dump(des, os.path.join(args["out"], fil + ".pkl"), compress=3)
