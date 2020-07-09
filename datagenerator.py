import numpy as np
from skimage import io, transform
import os
import tensorflow as tf
from keras.utils import to_categorical
import random

import pdb


class dataGenerator:
    'Generates data for Keras'
    def __init__(self, root, num_per_class = 3, dim=(224, 224,3),
                 n_classes=1, dataset='head'):
        'Initialization'
        self.root = root
        self.dim = dim
        self.batch_size = n_classes * num_per_class
        self.n_classes = n_classes
        self.num_per_class = num_per_class
        self.train_num = round(self.batch_size/3)*2

        if dataset == 'head':
            self.all_folders = [os.path.join(root, person) for person in os.listdir(root)]
        elif dataset == 'ominlog':
            self.all_folders = [os.path.join(root, family, character) \
                    for family in os.listdir(root) \
                    if os.path.isdir(os.path.join(root, family)) \
                    for character in os.listdir(os.path.join(root, family))]
        random.shuffle(self.all_folders)
        meta_num = 100
        self.meta_train_folders = self.all_folders[:meta_num]
        self.meta_test_folders = self.all_folders[meta_num:]


    def data_generation(self, train=True):
        '''Generates data containing batch_size samples'''
        # Initialization
        X = np.empty((self.batch_size, *self.dim), dtype='float32')
        Y = np.empty((self.batch_size), dtype = 'float32')
        if  train:
            folders = self.meta_train_folders
        else:
            folders = self.meta_test_folders

        paths = random.sample(folders, self.n_classes)
        # Generate data
        labels_and_images = self.get_images(paths=paths,
                                            labels=range(self.n_classes),
                                            nb_samples=self.num_per_class)
        for i, ID in enumerate(labels_and_images):
            # Store sample
            image = io.imread(ID[1])
            image = self.imgpreprocess(image)
            X[i,] = transform.resize(image, self.dim)
            Y[i,] = ID[0]
        Y = to_categorical(Y, self.n_classes)

        if train:
            inputa = X[:self.train_num, ]
            labela = Y[:self.train_num, ]
            inputb = X[self.train_num:, ]
            labelb = Y[self.train_num:, ]
            return inputa, labela, inputb, labelb
        else:
            return X, Y


    def imgpreprocess(self, rgb):
        rgb = rgb/255.0
        return rgb

    ## Image helper
    def get_images(self, paths, labels, nb_samples=None, shuffle=True):
        if nb_samples is not None:
            sampler = lambda x: random.sample(x, nb_samples)
        else:
            sampler = lambda x: x
        labels_and_images = [(i, os.path.join(path, image)) \
                  for i, path in zip(labels, paths) \
                  for image in sampler(os.listdir(path))]

        if shuffle:
            random.shuffle(labels_and_images)
        return labels_and_images


if __name__ == '__main__':
    root = './data/ominlog'
    generator = dataGenerator(root=root, n_classes=3, num_per_class=5, dim = (28,28,1))
    ia, la, ib, lb = generator.data_generation()
    pdb.set_trace()



