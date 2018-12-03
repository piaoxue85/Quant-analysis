import numpy as np
import pandas as pd
from cnn_lstm import DataGenerator, Preprocess
import keras
from keras.models import Sequential
from keras.layers import Dense,Activation,Softmax
from keras.layers import LSTM,Conv1D,Dropout,MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from data_processor import DataForModel as Data
from keras.utils import np_utils

data = DataGenerator(test =True)
data,label = Preprocess(data)
data = pd.concat([data,label],axis=1)

data = Data(data,test_ratio= 1)

x,y = data.get_test_batch(300,False)
y   = np_utils.to_categorical(y-1,num_classes=3)
print(x.shape,y.shape)

#Convolution
kernel_size = 5
pool_size = 2
time_step = 300
num_feature = 43
#LSTM
lstm_output_size = 32
batch_size = 32
epochs = 10

print('Build model...')
#Convolution
kernel_size = 5
pool_size = 2
time_step = 300
num_feature = 43
#LSTM
lstm_output_size = 32
batch_size = 32
epochs = 10

print('Build model...')
model = Sequential()
model.add(Conv1D(filters=16,input_shape = (time_step,num_feature),
                kernel_size = kernel_size,
                padding = 'causal',
                strides = 1))
model.add(PReLU())
model.add(BatchNormalization())
model.add(MaxPooling1D(2))
model.add(Conv1D(filters=16,
                kernel_size = kernel_size,
                padding = 'causal',
                strides = 1))
model.add(PReLU())
model.add(BatchNormalization())
model.add(MaxPooling1D(2))
model.add(Conv1D(filters=32,
                kernel_size = kernel_size,
                padding = 'causal',
                strides = 1))
model.add(PReLU())
model.add(BatchNormalization())
model.add(MaxPooling1D(2))
model.add(Conv1D(filters=32,
                kernel_size = kernel_size,
                padding = 'causal',
                strides = 1))
model.add(BatchNormalization())
model.add(MaxPooling1D(2))
model.add(LSTM(lstm_output_size))
model.add(Dense(units = 64))
model.add(PReLU())
model.add(Dropout(0.5))
model.add(Dense(units = 3))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

print('Train')
"""
model.fit(x_train,y_train,
          batch_size = batch_size,
          epochs = epochs,
          validation_data = (x_test,y_test))
          """
model.fit(x,y,
          batch_size = batch_size,
          epochs = epochs)
score, acc = model.evaluate(x,y, batch_size=batch_size)
print('Test cost:', score)
print('Test accuracy:', acc)
