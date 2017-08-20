include(macros.py.m4)

from keras.applications.vgg16 import preprocess_input
from tools import ImageDataGenerator, get_stratified_kfold, get_class_list, roundmult
import csv
import os
import numpy as np
from itertools import islice
from keras.utils.np_utils import to_categorical
from sklearn.externals import joblib
from random import shuffle
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau
import time

def init_net(nb_classes):
    # input image dimensions
    img_rows, img_cols = 224, 224
    if K.image_dim_ordering() == 'th':
        input_shape = (1, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, 3)
    # number of convolutional filters to use
    nb_filters = 16
    # size of pooling area for max pooling
    pool_size = (2, 2)
    # convolution kernel size
    kernel_size = (3, 3)

    model = Sequential()

    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid',
                            input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    return model

if __name__ == '__main__':
    GET_DATASET_INFO()
    # Initialiazing the model
    model = init_net(nb_classes=nb_classes)

    # Training params
    bat_size = 32
    nb_epoch = 100

    # Layers to be trained
    for layer in model.layers:
        layer.trainable = True

    from keras.optimizers import SGD as optimizer
    def compile(model):
        optim = optimizer(lr=10**-3)
        model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=["accuracy"])

    # Callbacks
    cb = list()
    cb.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                  patience=5, min_lr=10**-6))

    infos = lambda : {
        'model': 'train_lenet',
        'learning': {
            'optim': 'SGD',
            'optim_config': optim.get_config(),
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
