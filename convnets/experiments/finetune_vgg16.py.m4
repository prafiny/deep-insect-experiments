include(macros.py.m4)

from tools import ImageDataGenerator, roundmult
from keras.optimizers import SGD, RMSprop
import csv
import os
import numpy as np
from itertools import islice
from keras.utils.np_utils import to_categorical
from keras.layers import Input
from sklearn.externals import joblib
from random import shuffle
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Input
from keras.layers.pooling import GlobalAveragePooling2D
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
import time

if __name__ == "__main__":
    GET_DATASET_INFO()
    # Initialiazing the model
    input_size = (224, 224)
    base_model = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=get_input_shape(input_size))) 
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    predictions = Dense(nb_classes, activation="softmax", name="predictions")(x)

    # this is the model we will train
    model = Model(base_model.input, output=predictions)

    # Training params
    bat_size = 32
    nb_epoch = 50
    depth = `len'(model.layers)

    # Layers to be trained
    for layer in model.layers:
        layer.trainable = False
    for layer in model.layers[-depth:]:
        layer.trainable = True

    # The actual training
    from keras.optimizers import SGD
    def compile(model):
        optim = SGD(lr=10**-4, momentum=0.9)
        model.compile(optimizer=optim, loss="categorical_crossentropy", metrics=["accuracy"])

    # Callbacks
    cb = list()
    cb.append(ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                  patience=5, min_lr=10**-8))
    cb.append(ModelCheckpoint(filepath="vgg16_weights.hdf5",
                    verbose=1, save_best_only=True))

    infos = lambda : {
        "model": "vgg16",
        "learning": {
            "optim": "SGD",
            "optim_config": model.optimizer.get_config(),
            "weighting": "Yes" if args["with_weighting"] else "No",
            "history": res.history,
            "nb_epoch": nb_epoch,
            "depth": depth,
            "seed": None,
            "time": time.time() - start_time,
            "batch_size": bat_size,
            "image_db": {
                "nb_training": `len'(Xtrain),
                "nb_valid": `len'(Xvalid)
            }
        }
    }    

    TRAIN()
