#!/usr/bin/python3

'''Fairly basic set of tools for real-time data augmentation on image data.
Can easily be extended to include new transformations,
new preprocessing methods, etc...
'''
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import re
from scipy import linalg
import scipy.ndimage as ndi
from six.moves import range
import os
import threading
from PIL import Image

from keras import backend as K
from keras.preprocessing.image import Iterator, img_to_array, array_to_img, ImageDataGenerator


def flow_from_paths(self, filenames, labels,
                        target_size=(256, 256), color_mode='rgb',
                        classes=None, class_mode='categorical',
                        batch_size=32, shuffle=True, seed=None,
                        save_to_dir=None, save_prefix='', save_format='jpeg'):
    return PathsListIterator(
        filenames, labels, self,
        target_size=target_size, color_mode=color_mode,
        classes=classes, class_mode=class_mode,
        data_format=self.data_format,
        batch_size=batch_size, shuffle=shuffle, seed=seed,
        save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format)

ImageDataGenerator.flow_from_paths = flow_from_paths

def centered_crop(img, size):
    width, height = img.size
    new_width, new_height = size
    left = np.ceil((width - new_width)/2.)
    top = np.ceil((height - new_height)/2.)
    right = left + new_width
    bottom = top + new_height

    return img.crop((left, top, right, bottom))

def improved_resize(img, size):
    width = np.size(img, 1)
    height = np.size(img, 0)
    nsize = min(width, height)
    i = centered_crop(img, (nsize, nsize))
    i.thumbnail((size))
    return i

def load_img(path, grayscale=False, target_size=None):
    '''Load an image into PIL format.

    # Arguments
        path: path to image file
        grayscale: boolean
        target_size: None (default to original size)
            or (img_height, img_width)
    '''
    img = Image.open(path)
    if grayscale:
        img = img.convert('L')
    else:  # Ensure 3 channel even when loaded image is grayscale
        img = img.convert('RGB')
    if target_size:
        img = improved_resize(img, (target_size[1], target_size[0]))
        #img = img.resize((target_size[1], target_size[0]))
    return img

class PathsListIterator(Iterator):

    def __init__(self, filenames, labels, image_data_generator,
                 target_size=(256, 256), color_mode='rgb',
                 data_format=None,
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None,
                 save_to_dir=None, save_prefix='', save_format='jpeg'):
        if data_format is None:
            data_format = K.image_data_format()
        self.filenames = filenames
        self.labels = labels
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.data_format = data_format
        if self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        self.classes = classes
        if class_mode not in {'categorical', 'binary', 'sparse', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", or None.')
        self.class_mode = class_mode
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

        # first, count the number of samples and classes
        if not classes:
            classes = sorted(list(set(labels)))
        self.nb_class = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))
        self.nb_sample = len(labels)
        print('Found %d images belonging to %d classes.' % (self.nb_sample, self.nb_class))

        # second, build an index of the images in the different class subfolders
        self.classes = np.zeros((self.nb_sample,), dtype='int32')
        for i, cls in enumerate(labels):
            self.classes[i] = self.class_indices[cls]
        super(PathsListIterator, self).__init__(self.nb_sample, batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        batch_x = np.zeros((current_batch_size,) + self.image_shape)
        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            img = load_img(fname, grayscale=grayscale, target_size=self.target_size)
            x = img_to_array(img, data_format=self.data_format)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype('float32')
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), self.nb_class), dtype='float32')
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x
        return batch_x, batch_y
