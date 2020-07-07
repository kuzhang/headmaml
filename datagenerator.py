import keras
import numpy as np
from skimage import io, transform
import os
import random
import tensorflow as tf
from keras.utils import to_categorical
import math

class dataGeneratorv2(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, paths, num_per_class = 3, batch_size=5, dim=(224,224,3),
                 n_classes=1, shuffle=True, flags='train'):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.path = paths
        self.num_per_class = num_per_class
        self.flags=flags

        self.labels_and_images = self.get_images(paths=self.path,
                                            labels=range(self.n_classes),
                                            nb_samples=self.num_per_class)

        self.num_train = round(len(self.labels_and_images)/3)*2

        if self.flags=='train':
            self.meta_train_samples = self.labels_and_images[:self.num_train]
        elif self.flags=='test':
            self.meta_test_samples = self.labels_and_images[self.num_train:]


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(1)

    def __getitem__(self,index):
        'Generate one batch of data'
        # Generate indexes of the batch

        # Generate data
        if self.flags=='train':
            data_temp =self.meta_train_samples
        else:
            data_temp = self.meta_test_samples
        x, y = self.__data_generation(data_temp)

        return x, y

    def __data_generation(self, data_temp):
        '''Generates data containing batch_size samples'''
        # Initialization
        X = np.empty((len(data_temp), *self.dim), dtype='float32')
        Y = np.empty((len(data_temp)), dtype = 'float32')

        # Generate data
        for i, ID in enumerate(data_temp):
            # Store sample
            image = io.imread(ID[1])
            image = self.imgpreprocess(image)
            X[i,] = transform.resize(image, self.dim)
            Y[i,] = ID[0]
        Y = to_categorical(Y, self.n_classes)
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

class dataGeneratorv1:
    'Generates data for Keras'
    def __init__(self, paths, num_per_class = 3, dim=(224, 224,3),
                 n_classes=1, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = n_classes * num_per_class
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.path = paths
        self.num_per_class = num_per_class
        self.batches_num = math.floor(len(paths)/self.n_classes)
        self.train_num = round(self.batch_size/3)*2

    def data_generation(self):
        '''Generates data containing batch_size samples'''
        # Initialization
        X = np.empty((self.batch_size, *self.dim), dtype='float32')
        Y = np.empty((self.batch_size), dtype = 'float32')

        # Generate data
        labels_and_images = self.get_images(paths=self.path,
                                            labels=range(self.n_classes),
                                            nb_samples=self.num_per_class)
        for i, ID in enumerate(labels_and_images):
            # Store sample
            image = io.imread(ID[1])
            image = self.imgpreprocess(image)
            X[i,] = transform.resize(image, self.dim)
            Y[i,] = ID[0]
        Y = to_categorical(Y, self.n_classes)

        inputa = X[:self.train_num, ]
        labela = Y[:self.train_num, ]
        inputb = X[self.train_num:, ]
        labelb = Y[self.train_num:, ]
        return inputa, labela, inputb, labelb


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
    root = './data/'
    folders = [os.path.join(root, person) for person in os.listdir(root)]
    num_class = 5
    paths = random.sample(folders, num_class)
    print(len(paths))
    generator = dataGeneratorv1(paths=paths, n_classes=5)



