'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

batch_size = 10
nb_classes = 2
nb_epoch = 18

# input image dimensions
time, freq = 1292, 128
# number of convolutional filters to use
nb_filters1 = 256
nb_filters2 = 512
# size of pooling area for max pooling
nb_pool = 4
# convolution kernel size
nb_conv = 4
nn_layer = 2048

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution1D(nb_filters1, nb_conv, 
    border_mode='same',
    input_shape=(1, time, freq)))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=4))
model.add(Dropout(0.25))


model.add(Convolution1D(nb_filters1, nb_conv, 
    border_mode='same',
    ))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.25))

model.add(Convolution1D(nb_filters2, nb_conv, 
    border_mode='same',
    ))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.25))


def global_pooling(x):
    m = K.mean(x, axis=1, keepdims=True)
    l2 = K.l2_normalize(x, axis=1, keepdims=True)
    x_max = K.max(x, axis=1, keepdims=True)
    return K.concatenate([m, l2, x_max], axis=1)

def gp_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 2  # only valid for 2D tensors
    size = tuple(shape[0]*3, 1)
    return size

#model.add(Lambda(global_pooling, output_shape=gp_output_shape))
model.add(Lambda(global_pooling))

model.add(Dense(nn_layer))
model.add(Dense(nn_layer))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy',
        optimizer=sgd,
        metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
        verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
