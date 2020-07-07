import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as keras_backend
from tensorflow.keras.layers import Flatten, Input
from keras import models,optimizers


# mine define methods
def generate_dataset(generator, num_class, num_per_class):
    '''
    Generate train and test dataset.
    '''
    total_samples = num_per_class * num_class
    print('total samples for meta training:',total_samples)
    train_num = round(total_samples/3)*2
    x, y = generator.data_generation()
    inputa = x[:train_num,]
    labela = y[:train_num,]
    inputb = x[train_num:,]
    labelb = y[train_num:,]
    return inputa, labela, inputb, labelb


class CNNModel(tf.keras.Model):

    def __init__(self, num_class=1, inputshape=(224,224,3)):
        super(CNNModel, self).__init__()
        self.conv_base = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=inputshape)
        for layer in self.conv_base.layers[:-4]:
           layer.trainable = False
        self.ft = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(4096, activation='relu')
        self.do1 = tf.keras.layers.Dropout(0.5)
        self.fc2 = tf.keras.layers.Dense(4096, activation='relu')
        self.do2 = tf.keras.layers.Dropout(0.5)
        self.fc3 = tf.keras.layers.Dense(num_class, activation='sigmoid')

    def forward(self, x):
        x = self.conv_base(x)
        x = self.ft(x)
        x = self.fc1(x)
        x = self.do1(x)
        x = self.fc2(x)
        x = self.do2(x)
        z = self.fc3(x)
        return z

def CNNModelv3(num_class=1, inputshape=(224,224,3)):

        input = Input(shape = inputshape)
        conv_base = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=inputshape)
        for layer in conv_base.layers[:-4]:
           layer.trainable = False
        x = conv_base(input)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(4096, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(4096, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(num_class, activation='sigmoid')(x)
        model = tf.keras.Model(input, x)
        return model




def CNNModelv2(num_class=1):
    conv_base = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
    for layer in conv_base.layers[:-4]:
       layer.trainable = False
    model=tf.keras.Sequential()
    model.add(conv_base)
    model.add(Flatten())
    model.add(tf.keras.layers.Dense(4096, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(4096, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(num_class, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    return model
