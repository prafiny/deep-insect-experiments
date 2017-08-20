#!/usr/local/bin/python2.7
#-*- encoding:utf-8 -*-

import argparse as ap
import cv2
import imutils 
import csv
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.multiclass import OneVsRestClassifier

# Load the classifier, class names, scaler, number of clusters and vocabulary 
clf, classes_names, stdSlr, k, voc = joblib.load("bof.pkl")

# Get the path of the testing set
parser = ap.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-t", "--testingSet", help="Path to testing Set")
group.add_argument("-i", "--image", help="Path to image")
parser.add_argument('-v',"--visualize", action='store_true')
parser.add_argument("-c", "--csv")
args = vars(parser.parse_args())

# Get the path of the testing image(s) and store them in a list
image_paths = []
if args["testingSet"]:
    test_path = args["testingSet"]
    try:
        testing_names = os.listdir(test_path)
    except OSError:
        print "No such directory {}\nCheck if the file exists".format(test_path)
        exit()
    if args["csv"]:
        with open(args['csv'], 'r') as csvfile:
            r = csv.reader(csvfile, delimiter=',')
            csvcontent = [[unicode(cell, 'utf-8') for cell in row] for row in r]
            image_paths = [os.path.join(test_path, f[0]) for f in csvcontent]
    else:
        for testing_name in testing_names:
            dir = os.path.join(test_path, testing_name)
            class_path = imutils.imlist(dir)
            image_paths+=class_path
elif args["image"]:
    image_paths = [args["image"]]
    
# Create feature extraction and keypoint detector objects
fea_det = cv2.FeatureDetector_create("SIFT")
des_ext = cv2.DescriptorExtractor_create("SIFT")

# List where all the descriptors are stored
des_list = []

for image_path in image_paths:
    im = cv2.imread(image_path)
    if im.shape is None:
        print "No such file {}\nCheck if the file exists".format(image_path)
        exit()
    kpts = fea_det.detect(im)
#    kp_img = im.copy()
#    cv2.drawKeypoints(im, kpts, kp_img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#    imutils.show(kp_img)
    kpts, des = des_ext.compute(im, kpts)
    des_list.append((image_path, des))   
    
# Stack all the descriptors vertically in a numpy array
descriptors = des_list[0][1]
for image_path, descriptor in des_list[0:]:
    descriptors = np.vstack((descriptors, descriptor)) 

# 
test_features = np.zeros((len(image_paths), k), "float32")
for i in xrange(len(image_paths)):
    words, distance = vq(des_list[i][1],voc)
    for w in words:
        test_features[i][w] += 1

# Perform Tf-Idf vectorization
nbr_occurences = np.sum( (test_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

# Scale the features
test_features = stdSlr.transform(test_features)

for f, d in zip(image_paths, test_features):
    print(os.path.basename(f) + '\t' + "\t".join([str(i) for i in d]))
