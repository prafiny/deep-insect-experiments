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
from sklearn.model_selection import GridSearchCV
from train import get_features

# Get the path of the testing set
parser = ap.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-t", "--testingSet", help="Path to testing Set")
group.add_argument("-i", "--image", help="Path to image")
parser.add_argument('-v',"--visualize", action='store_true')
parser.add_argument("-c", "--csv")
parser.add_argument("-m", "--model", default="bof.pkl")

args = vars(parser.parse_args())

# Load the classifier, class names, scaler, number of clusters and vocabulary 
clf, classes_names, stdSlr, k, voc = joblib.load(args["model"])

# Get the path of the testing image(s) and store them in a list
image_paths = []
if args["testingSet"]:
    test_path = args["testingSet"]
    try:
        testing_names = os.listdir(test_path)
    except OSError:
        print("No such directory {}\nCheck if the file exists".format(test_path))
        exit()
    if args["csv"]:
        with open(args['csv'], 'r') as csvfile:
            r = csv.reader(csvfile, delimiter=',')
            csvcontent = [[cell for cell in row] for row in r]
            image_paths = [f[0] for f in csvcontent]
    else:
        for testing_name in testing_names:
            dir = os.path.join(test_path, testing_name)
            class_path = imutils.imlist(dir)
            image_paths+=class_path
elif args["image"]:
    image_paths = [args["image"]]
    

test_features = (get_features(test_path, [image_path], stdSlr, voc, k) for image_path in image_paths)
# Perform the predictions
decision_func = (clf.decision_function(p)[0] for p in test_features)
top = ([c for c, _ in sorted(list(zip(classes_names, r)), key=lambda t: t[1], reverse=True)] for r in decision_func)

# Visualize the results, if "visualize" flag set to true by the user
if args["visualize"]:
    for image_path, prediction in zip(image_paths, predictions):
        image = cv2.imread(image_path)
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        pt = (0, 3 * image.shape[0] // 4)
        cv2.putText(image, prediction, pt ,cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 2, [0, 255, 0], 2)
        cv2.imshow("Image", image)
        cv2.waitKey(3000)
else:
    print("\n".join(",".join(t) for t in zip([os.path.basename(i) for i in image_paths], [",".join(c) for c in top])))
