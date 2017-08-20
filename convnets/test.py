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
get_name = lambda x: classes_names[x]
nb_classes = len(classes_names)

def decode_predictions(res):
    return list(sorted(zip(classes_names, list(res)), key=lambda x: x[1], reverse=True))

# Get the path of the testing image(s) and store them in a list
with open(args['csv'], 'r') as csvfile:
    r = csv.reader(csvfile, delimiter=',')
    csvcontent = [[cell for cell in row] for row in r]
    image_paths = [os.path.join(args['image_folder'], f[0]) for f in csvcontent]

model = load_model(args["model"])

for img_path in image_paths:
    img = load_img(img_path, target_size=(224, 224))
    x = prepare_image(img)
    preds = model.predict(x)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    print(os.path.basename(img_path) + ',' + ','.join([n for n, _ in decode_predictions(preds[0])]))
