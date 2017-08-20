include(macros.py.m4)

from tools import ImageDataGenerator, get_class_list, roundmult
import csv
import os
import numpy as np
from itertools import islice
from keras.utils.np_utils import to_categorical
from keras.layers import Input
from sklearn.externals import joblib
from random import shuffle
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Input
from keras.layers.pooling import GlobalAveragePooling2D
from keras.callbacks import ReduceLROnPlateau
import time

if __name__ == "__main__":
    GET_DATASET_INFO()
    # Initialiazing the model
    input_size = (224, 224) 
    base_model = ResNet50(weights=None, include_top=False, input_tensor=Input(shape=get_input_shape(input_size))) 
    x = base_model.output
    x = Flatten()(x)
    predictions = Dense(nb_classes, activation='softmax', name='predictions')(x)

    model = Model(base_model.input, output=predictions)

    # Training params
    bat_size = 32
    nb_epoch = 200

    # Layers to be trained
    for layer in model.layers:
        layer.trainable = True

    from keras.optimizers import SGD

    def compile(model):
        optim = SGD(lr=10**-1)
        model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=["accuracy"])

    # Callbacks
    cb = list()
    cb.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                  patience=5, min_lr=10**-6))

    infos = lambda : {
        'model': 'resnet50',
        'learning': {
            'optim': 'SGD',
            'optim_config': model.optimizer.get_config(),
            'history': res.history,
            'nb_epoch': nb_epoch,
            'seed': None,
            'time': time.time() - start_time,
            'batch_size': bat_size,
            'image_db': {
                'nb_training': `len'(Xtrain),
                'nb_valid': `len'(Xvalid)
            }
        }
    }    

    TRAIN()
