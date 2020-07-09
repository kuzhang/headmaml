import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Flatten, Input
from tensorflow.keras.initializers import RandomNormal
import pdb


def CNNModelv1(num_class=1, inputshape=(224,224,3), backbone = 'vgg16'):

        input = Input(shape = inputshape)
        x =[]
        if backbone == 'vgg16':
            conv_base = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=inputshape)
            for layer in conv_base.layers:
               layer.trainable = False
            x = conv_base(input)
            x = tf.keras.layers.Flatten()(x)
        elif backbone == 'resnet50':
            conv_base = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=inputshape)
            for layer in conv_base.layers[:-4]:
                layer.trainable = False
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            print(x)
            pdb.set_trace()
        x = tf.keras.layers.Dense(4096, activation='relu', kernel_initializer=RandomNormal())(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(4096, activation='relu',kernel_initializer=RandomNormal())(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(num_class, activation='sigmoid',kernel_initializer=RandomNormal())(x)
        model = tf.keras.Model(input, x)
        return model


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, MaxPooling2D
weight_init = RandomNormal()

def CNNModelv2(num_class=1, inputshape=(28,28,1)):
    # Build model
    model = Sequential()
    model.add(
        Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer=weight_init, input_shape=inputshape))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer=weight_init))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer=weight_init))
    model.add(Dropout(0.5))
    model.add(Dense(num_class, activation='softmax', kernel_initializer=weight_init))

    return model