import numpy as np
import os
os.environ['KERAS_BACKEND']='tensorflow'
import keras
from keras.layers import Input,Dense,LSTM,TimeDistributed,Activation,Dropout,Conv1D,BatchNormalization,Masking,Bidirectional,MaxPooling1D
from keras.models import Model, Sequential
from keras.layers.recurrent import SimpleRNN
from keras.layers.core import RepeatVector

import data_utils
import argparse


############################################
#data preprocessing
parser = argparse.ArgumentParser()

parser.add_argument('--train_path', type=str, default="./data/mfcc/train.ark", 
                    help='training data directory')
parser.add_argument('--test_path', type=str, default="./data/mfcc/test.ark", 
                    help='testing data directory')

parser.add_argument('--label_path', type=str, default="./data/label/train.lab", 
                    help='training label directory')
parser.add_argument('--testlabelmap', type=str, default="./data/label/testlabelmap.txt", 
                    help='label number map directory')
parser.add_argument('--map48_dir', type=str, default="./data/phones/48phone_char.map", 
                    help='48 phones and ENG letters directory')
parser.add_argument('--map48to39_dir', type=str, default="./data/phones/48_39.map", 
                    help='48 convert 39 phones directory')

args = parser.parse_args()


X, Y = data_utils.load_train_data(args.train_path, args.label_path,
                                             args.map48_dir, args.map48to39_dir, args.testlabelmap)

#################################################
#model rnn


timesteps = 777
features = 39

model = Sequential()

model.add(Masking(mask_value=0.0, input_shape=(777, 39)))

model.add(BatchNormalization())

##
#model.add(Bidirectional(LSTM(units=128,activation='relu', return_sequences = True)))
#model.add(Bidirectional(LSTM(units=256, return_sequences = True)))
model.add(Bidirectional(LSTM(units=256, return_sequences = True)))

model.add(Bidirectional(LSTM(units=256, return_sequences = True)))
#model.add(Dropout(0.25))
#model.add(BatchNormalization())

model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(TimeDistributed(Dense(40)))

model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

model.fit(x = X, y = Y, batch_size = 25, epochs = 5, verbose=1, validation_split = 0.1)
#
#
model.save('my_model1027rnn.h5')  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model




