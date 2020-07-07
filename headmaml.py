import tensorflow as tf
# Other dependencies
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import os, sys

# self define functions and classes
from datagenerator import dataGeneratorv1
from utils import *
import pdb


# Reproduction
np.random.seed(333)
print('Python version: ', sys.version)
print('TensorFlow version: ', tf.__version__)

# Data generation
root = r'C:\Users\kkdez\Desktop\dataset\rgbhead\sequence1to3'
folders = [os.path.join(root, person) for person in os.listdir(root)]
num_class = 3
num_per_class =5
input_shape = (112, 112, 3)

#paths = random.sample(folders, num_class)
generator = dataGeneratorv1(paths=folders,
                          num_per_class=num_per_class,
                          n_classes=num_class,
                          dim = input_shape)
#inp = generate_dataset(generator, num_class, num_per_class)

# create model
model = CNNModelv3(num_class, input_shape)

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

            with tf.GradientTape() as test_tape:
                # Step 5
                for _ in range(steps):
                    with tf.GradientTape() as train_tape:
                        preda = model(inputa)
                        train_loss = tf.keras.losses.categorical_crossentropy(labela, preda)
                    print('train loss:{}'.format(train_loss))
                    gradients = train_tape.gradient(train_loss, model.trainable_variables)
                    print('train gradient', gradients[-1].numpy())
                    # Update model
                    inner_optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                # Step 8
                predb = model(inputb)
                test_loss = tf.keras.losses.categorical_crossentropy(labelb, predb)

            # Step 10
            #pdb.set_trace()
            model.set_weights(old_weights)
            gradients = test_tape.gradient(test_loss, model.trainable_variables)
            outer_optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            print('test test gradient', gradients[-1].numpy())
            print('test weights update', model.trainable_variables[-1].numpy())
            print('test loss:',test_loss.numpy())

train_maml(model, generator, 5)
model.compile(optimizer=outer_optimizer, loss=tf.keras.losses.categorical_crossentropy, metrics=['acc']) # Compile just for evaluation