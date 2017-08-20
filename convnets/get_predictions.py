import argparse as ap
parser = ap.ArgumentParser()
parser.add_argument("--image-folder", required=True)
parser.add_argument("-c", "--csv", required=True)
parser.add_argument("-m", "--model", default="model.h5")
args = vars(parser.parse_args())

import csv
import numpy as np
import os
from tools import load_img
from keras.preprocessing import image
from sklearn.externals import joblib
from keras.models import load_model
from random import random
import sys


def get_class_dict(path):
	with open(path, newline='') as f:
		r = csv.reader(f)
		return {l[0]:l[1] for l in r}

def get_class_list(class_dict):
	return set(class_dict.values())

from matplotlib import colors as mcolors
from random import shuffle
colors = ["#" + c for c in ["FF0000", "00FF00", "0000FF", "FFFF00", "FF00FF", "00FFFF", "000000", 
        "800000", "008000", "000080", "808000", "800080", "008080", "808080", 
        "C00000", "00C000", "0000C0", "C0C000", "C000C0", "00C0C0", "C0C0C0", 
        "400000", "004000", "000040", "404000", "400040", "004040", "404040", 
        "200000", "002000", "000020", "202000", "200020", "002020", "202020", 
        "600000", "006000", "000060", "606000", "600060", "006060", "606060", 
        "A00000", "00A000", "0000A0", "A0A000", "A000A0", "00A0A0", "A0A0A0", 
        "E00000", "00E000", "0000E0", "E0E000", "E000E0", "00E0E0", "E0E0E0"]]
shuffle(colors)

def gen_color_dict(class_list):
	return dict(zip(class_list, colors))

def from_categorical(x):
    return np.argmax(x, axis=1)[0]

mean, std = joblib.load("meanstd.pkl")

def prepare_image(img):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x -= mean
    x = np.divide(x, std)
    return x

classes_names = joblib.load("training_names.pkl")
color_dict = gen_color_dict(classes_names)
get_name = lambda x: classes_names[x]
nb_classes = len(classes_names)

def decode_predictions(res):
    return list(sorted(zip(classes_names, list(res)), key=lambda x: x[1], reverse=True))

# Get the path of the testing image(s) and store them in a list
with open(args['csv'], 'r') as csvfile:
    r = csv.reader(csvfile, delimiter=',')
    csvcontent = [[cell for cell in row] for row in r]

image_paths = [os.path.join(args['image_folder'], f[0]) for f in csvcontent]
image_classes = [f[1] for f in csvcontent]


model = load_model(args["model"])

for img_path, img_class in zip(image_paths, image_classes):
    img = load_img(img_path, target_size=(224, 224))
    x = prepare_image(img)
    preds = model.predict(x)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    infos = "{};class={};color={}".format(os.path.basename(img_path), img_class, color_dict[img_class])
    print(infos + '\t' + '\t'.join(["{}".format(n) for n in list(preds[0])]))
