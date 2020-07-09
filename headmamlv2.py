import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from datagenerator import dataGenerator
from utils import *

# Create Data generator
root = r'C:\Users\kkdez\Desktop\dataset\rgbhead\sequence1to3'
#root = './data/ominlog/'
num_class = 10
num_per_class =10
input_shape = (14, 14, 3)
generator = dataGenerator(root=root,
                          num_per_class=num_per_class,
                          n_classes=num_class,
                          dim = input_shape,
                          dataset='head')

# Hyperparameters
epochs = 10
inner_optimizer = Adam(lr=0.1)
outer_optimizer = Adam(lr=0.1)
weight_init = RandomNormal()

# Build model
model = CNNModelv2(num_class, input_shape)

# Meta training
def step(train_x, train_y, test_x, test_y ):
    old_weights = model.get_weights()
    #print('old weights', old_weights[-1])

    # step 6 in algorithm 2
    with tf.GradientTape() as tape:
        # Make prediction
        pred_y = model(train_x)
        # Calculate loss
        model_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(train_y, pred_y))
    # step 7 in algorithm 2
    # Calculate gradients
    model_gradients = tape.gradient(model_loss, model.trainable_variables)
    # Update model
    inner_optimizer.apply_gradients(zip(model_gradients, model.trainable_variables))
    print('train loss:', tf.keras.backend.mean(model_loss).numpy())
    print('train gradient', model_gradients[-1].numpy())
    #print('model trianable variables:', model.trainable_variables[-1].numpy())

    #step 10 in algorithm 2
    with tf.GradientTape() as test_tape:
        pred_y = model(test_x)
        # Calculate loss
        test_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(test_y, pred_y))
    model.set_weights(old_weights)
    test_gradients = test_tape.gradient(test_loss, model.trainable_variables)
    # Update model
    outer_optimizer.apply_gradients(zip(test_gradients, model.trainable_variables))
    print('test loss:', tf.keras.backend.mean(test_loss).numpy())
    print('test gradient', test_gradients[-1].numpy())
    #print('test model trianable variables:', model.trainable_variables[-1].numpy())


# Training loop
for epoch in range(epochs):
    print('epoch', epoch)
    # step 5 & 8 in algorithm 2
    inputa, labela, inputb, labelb = generator.data_generation(train=True)
    step(inputa, labela, inputb, labelb)

# Calculate accuracy
model.compile(optimizer=outer_optimizer, loss=tf.keras.losses.categorical_crossentropy, metrics=['acc']) # Compile just for evaluation
x_test, y_test = generator.data_generation(train=False)
print('\n', model.evaluate(x_test, y_test, verbose=0)[1])