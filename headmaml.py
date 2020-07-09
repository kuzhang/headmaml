import tensorflow as tf
# Other dependencies
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import os, sys

# self define functions and classes
from datagenerator import dataGenerator
from utils import *
import pdb


# Reproduction
np.random.seed(333)
print('Python version: ', sys.version)
print('TensorFlow version: ', tf.__version__)

# Data generation
#root = r'C:\Users\kkdez\Desktop\dataset\rgbhead\sequence1to3'
root = './data/ominlog/'

num_class = 2
num_per_class =5
input_shape = (28, 28, 1)

generator = dataGenerator(root=root,
                          num_per_class=num_per_class,
                          n_classes=num_class,
                          dim = input_shape,
                          dataset='ominlog')
#inp = generate_dataset(generator, num_class, num_per_class)

# create model
#model = CNNModelv1(num_class, input_shape, 'vgg16')
model = CNNModelv2(num_class, inputshape=input_shape)

# define maml training parameters
steps = 1
lr_update = 10
batch_num =10
inner_optimizer = tf.keras.optimizers.Adam(lr=0.1)
outer_optimizer = tf.keras.optimizers.Adam(lr=0.1)


def train_maml(model, generator, epochs, batch_size=1, log_steps=1000):
    '''Train using the MAML setup.

    The comments in this function that start with:

        Step X:

    Refer to a step described in the Algorithm 1 of the paper.

    Args:
        model: A model.
        epochs: Number of epochs used for training.
        dataset: A dataset used for training.
        lr_inner: Inner learning rate (alpha in Algorithm 1). Default value is 0.01.
        batch_size: Batch size. Default value is 1. The paper does not specify
            which value they use.
        log_steps: At every `log_steps` a log message is printed.

    Returns:
        A strong, fully-developed and trained maml.
    '''
    # Step 2: instead of checking for convergence, we train for a number
    # of epochs
    for epoch in range(epochs):
        total_loss = 0
        losses = []
        start = time.time()
        print('epoch:', epoch)
        # Step 3 and 4
        for batch in range(batch_num):
            print('bacth number:', batch)
            inputa, labela, inputb, labelb = generator.data_generation()
            old_weights = model.get_weights()
            print('old weights:{}'.format(model.trainable_variables[-1].numpy()))


            # Step 5
            for _ in range(steps):
                with tf.GradientTape(persistent=True) as train_tape:
                    preda = model(inputa)
                    train_loss = tf.keras.losses.categorical_crossentropy(labela, preda)
                train_gradients = train_tape.gradient(train_loss, model.trainable_variables)
                # Update model
                inner_optimizer.apply_gradients(zip(train_gradients, model.trainable_variables))
                """
                for j in range(len(model.trainable_variables)):
                    delta = tf.subtract(model.trainable_variables[j], tf.multiply(0.1, gradients[j])).numpy()
                    model.trainable_variables[j].assign(delta)
                """
                print('train loss:{}'.format(train_loss))
                print('train gradient', train_gradients[-1].numpy())
                print('train weights update',model.trainable_variables[-1].numpy())

            # Step 8
            with tf.GradientTape(persistent=True) as test_tape:
                predb = model(inputb)
                test_loss = tf.keras.losses.categorical_crossentropy(labelb, predb)

            # Step 10
            #pdb.set_trace()
            #model.set_weights(old_weights)
            test_gradients = test_tape.gradient(test_loss, model.trainable_variables)
            outer_optimizer.apply_gradients(zip(test_gradients, model.trainable_variables))

            print('test gradient', test_gradients[-1].numpy())
            print('test weights update', model.trainable_variables[-1].numpy())
            print('test loss:',test_loss.numpy())

train_maml(model, generator, 5)
pdb.set_trace()
model.compile(optimizer=outer_optimizer, loss=tf.keras.losses.categorical_crossentropy, metrics=['acc']) # Compile just for evaluation