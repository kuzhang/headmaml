import tensorflow as tf
import math
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal

# Load and pre-process training data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = (x_train / 255).reshape((-1, 28, 28, 1))
y_train = tf.keras.utils.to_categorical(y_train, 10)
x_test = (x_test / 255).reshape((-1, 28, 28, 1))
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Hyperparameters
batch_size = 128
epochs = 1000
inner_optimizer = Adam(lr=0.001)
outer_optimizer = Adam(lr=0.001)
weight_init = RandomNormal()

from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, MaxPooling2D
from tensorflow.keras.models import Sequential

# Build model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer=weight_init, input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer=weight_init))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer=weight_init))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax', kernel_initializer=weight_init))


def step(real_x, real_y):
    train_x = real_x[:64,]
    train_y = real_y[:64,]
    test_x = real_x[64:,]
    test_y = real_y[64:,]
    old_weights = model.get_weights()
    #print('old weights', old_weights[-1])
    with tf.GradientTape() as tape:
        # Make prediction
        pred_y = model(train_x.reshape((-1, 28, 28, 1)))
        # Calculate loss
        model_loss = tf.keras.losses.categorical_crossentropy(train_y, pred_y)

    #print('model loss:', model_loss)
    # Calculate gradients
    model_gradients = tape.gradient(model_loss, model.trainable_variables)
    #print('model gradient', model_gradients[-1].numpy())
    # Update model
    inner_optimizer.apply_gradients(zip(model_gradients, model.trainable_variables))
    #print('model trianable variables:', model.trainable_variables[-1].numpy())

    with tf.GradientTape() as test_tape:
        pred_y = model(test_x.reshape((-1, 28, 28, 1)))
        # Calculate loss
        test_loss = tf.keras.losses.categorical_crossentropy(test_y, pred_y)
    model.set_weights(old_weights)
    model_gradients = test_tape.gradient(test_loss, model.trainable_variables)
    # print('model gradient', model_gradients[-1].numpy())
    # Update model
    outer_optimizer.apply_gradients(zip(model_gradients, model.trainable_variables))
    print('test loss:', tf.keras.backend.mean(test_loss).numpy())
    #print('test gradient', model_gradients[-1])
    #print('test model trianable variables:', model.trainable_variables[-1].numpy())


# Training loop
bat_per_epoch = math.floor(len(x_train) / batch_size)
for epoch in range(epochs):
    print('epoch', epoch)
    #for i in range(bat_per_epoch):
    i = 0
    n = i*batch_size
    step(x_train[n:n+batch_size], y_train[n:n+batch_size])

# Calculate accuracy
model.compile(optimizer=outer_optimizer, loss=tf.keras.losses.categorical_crossentropy, metrics=['acc']) # Compile just for evaluation
print('\n', model.evaluate(x_test, y_test, verbose=0)[1])