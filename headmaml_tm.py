import tensorflow as tf
import tensorflow.keras as keras

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


def copy_model(model, x):
    '''Copy model weights to a new model.

    Args:
        model: model to be copied.
        x: An input example. This is used to run
            a forward pass in order to add the weights of the graph
            as variables.
    Returns:
        A copy of the model.
    '''
    copied_model = CNNModel(num_class=3)

    # If we don't run this step the weights are not "initialized"
    # and the gradients will not be computed.
    copied_model.forward(tf.convert_to_tensor(x))

    copied_model.set_weights(model.get_weights())
    return copied_model

root = r'C:\Users\kkdez\Desktop\dataset\rgbhead\sequence1to3'
folders = [os.path.join(root, person) for person in os.listdir(root)]
num_class = 3
num_per_class =20

paths = random.sample(folders, num_class)
generator = dataGeneratorv1(paths=paths,
                          num_per_class=num_per_class,
                          n_classes=num_class)
inputa, labela, inputb, labelb = generate_dataset(generator, num_class, num_per_class)

model = CNNModel(num_class)
#copy_model = CNNModel(num_class)

cce = keras.losses.CategoricalCrossentropy()
steps = 1
lr_update = 10


def train_maml(model, epochs, inputa, labela, lr_inner=0.01, batch_size=1, log_steps=1000):
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
    optimizer = keras.optimizers.Adam()

    # Step 2: instead of checking for convergence, we train for a number
    # of epochs
    for _ in range(epochs):
        total_loss = 0
        losses = []
        start = time.time()
        # Step 3 and 4

        x, y = inputa, labela
        model.forward(x)  # run forward pass to initialize weights


        with tf.GradientTape() as test_tape:

            test_tape.watch(model.trainable_variables)
            # Step 5
            with tf.GradientTape() as train_tape:
                train_loss, _ = compute_loss(model, x, y)
            # Step 6
            gradients = train_tape.gradient(train_loss, model.trainable_variables)
            
            model_copy = copy_model(model, x)

            for j in range(len(model_copy.trainable_variables)):
                model_copy.trainable_variables[j] = tf.subtract(model.trainable_variables[j],
                                                                    tf.multiply(lr_inner, gradients[j]))
            # Step 8
            test_loss, logits = compute_loss(model_copy, x, y)

            # Step 10
        gradients = test_tape.gradient(test_loss, model.trainable_variables)
        #pdb.set_trace()
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        #print(test_loss)
        
        total_loss += test_loss
        


train_maml(model,1, inputa, labela)