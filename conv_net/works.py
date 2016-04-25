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
from keras.layers.core import Dense, Dropout, Activation, Flatten, Lambda, Reshape
from keras.layers.convolutional import Convolution1D, MaxPooling1D, AveragePooling1D
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split
from keras import backend as K

batch_size = 10
nb_classes = 3
nb_epoch = 100

# input image dimensions
time, freq = 1292, 128
# number of convolutional filters to use
nb_filters1 = 256
nb_filters2 = 512
# size of pooling area for max pooling
nb_pool = 8
# convolution kernel size
nb_conv = 4
nn_layer = 30

# the data, shuffled and split between train and test sets
#(X_train, y_train), (X_test, y_test) = mnist.load_data()
X = np.load('../../numpy_data/X.npy')
Y = np.load('../../numpy_data/Y.npy')
print(Y.shape)
#Y[100:,0] = 1
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print(y_train)
print(y_train.shape)
#X_train = X_train.reshape(X_train.shape[0], 1, time, freq)
#X_test = X_test.reshape(X_test.shape[0], 1, time, freq)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
#X_train /= 255
#X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train.astype(int), nb_classes)
Y_test = np_utils.to_categorical(y_test.astype(int), nb_classes)

print ('Y_train shape:', Y_train.shape)
model = Sequential()

model.add(Convolution1D(nb_filters1, nb_conv, 
    border_mode='valid',
    input_shape=(time, freq)))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_length=nb_pool))
model.add(Dropout(0.75))


model.add(Convolution1D(nb_filters1, nb_conv, 
    border_mode='valid',
    ))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_length=nb_pool/2))
#model.add(Dropout(0.25))

#model.add(Convolution1D(nb_filters2, nb_conv, 
#    border_mode='valid',
#    ))
#model.add(Activation('relu'))
#model.add(MaxPooling1D(pool_length=nb_pool/2))
#model.add(Dropout(0.25))


def global_pooling(x):
    m = K.mean(x, axis=1)
    return m
    #x_max = K.max(x, axis=1)
    #l2 = K.l2_normalize(x, axis=1)
    #x_max = K.max(x, axis=2)
    #print(K.shape(m), K.shape(l2), K.shape(x_max))
    #return K.concatenate([m, x_max], axis=1)
    #return m

def gp_output_shape(input_shape):
    shape = list(input_shape)
    print(shape)
    #assert len(shape) == 2  # only valid for 2D tensors
    size = (None, shape[2])
    return size

print(model.summary())

#model.add(K.concatenate[temp1, temp2], axis=-1)
model.add(Lambda(global_pooling, output_shape=gp_output_shape))
#model.add(Reshape((1536,)))
#model.add(K.m
#model.add(Lambda(global_pooling))
#model.add(Flatten())
model.add(Dense(nn_layer))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
#model.add(Activation('linear'))
model.add(Activation('softmax'))
print(model.summary())
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy',
        optimizer='adadelta',
        metrics=['accuracy'])

#model.compile(loss='categorical_crossentropy',
#        optimizer=sgd,
#        metrics=['accuracy'])
#model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
#        verbose=1, validation_data=(X_test, Y_test))

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
        verbose=1, validation_split=0.1)
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
