import os
import numpy as np
import tensorflow as tf
import shutil
import glob as glob
import cv2
import matplotlib.pyplot as plt
import random


from keras.models import load_model
model = load_model('/home/zongbangtan/conv_LSTM/autoencoder-sigmoid/Reconstruction_model_5/autoencoder.hdf5')
layer_name = 'average_pooling2d_2'
encoder = tf.keras.Model(inputs=model.input,
                                       outputs=model.get_layer(layer_name).output)
layer_name = 'conv2d_6'
decoder = tf.keras.Model(inputs=encoder.output,outputs=model.get_layer(layer_name).output)

#dataset an example
mobility = ['M=1.60', 'M=1.55', 'M=1.05', 'M=1.45', 'M=1.40', 'M=1.35','M=1.30', 'M=1.25', 'M=1.20', 'M=1.15']
layer = ['layer_1','layer_39','layer_79','layer_100','layer_69','layer_90','layer_110','layer_120','layer_50','layer_60']
train_images = []
from tkinter import Tcl
from PIL import Image
for mob in mobility:
    for lay in layer:
        X = glob.glob(f"/home/zongbangtan/data_for_lstm/gamma=50/{mob}/sliced_by_layer/{lay}/*.jpg")
        X_sorted = Tcl().call('lsort', '-dict', X)
        print(len(X_sorted))
for i,path in enumerate(X_sorted[0:]):
  image = cv2.imread(X_sorted[i], cv2.IMREAD_GRAYSCALE)
  img = np.array(image)
  # img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)
  img = img.astype("float32") / 255
  train_images.append(img)

print(len(train_images))
# 1k images

# Converting to a numpy array
train_images = np.array(train_images)
print(train_images.shape)

X_train = train_images
print(X_train.shape)

X_train.shape
encoded_images = encoder.predict(X_train)

print(encoded_images.shape)

convlstm_input = []

for i in range(0,9940):
  temp = encoded_images[i:i+60,...]  #选取前40帧图像
  convlstm_input.append(temp)

convlstm_input = np.array(convlstm_input)

# Input is a 5D array - (samples, time_steps, H,W,filters)
convlstm_input.shape

# Inputs for the ConvLSTM model
X = convlstm_input

# Y will be the current 40frames + 1st image
Y = encoded_images[60:,...]

print(X.shape)
print(Y.shape)
tf.keras.backend.clear_session()
inp = tf.keras.layers.Input(shape=(None, 16,16,4))

x = tf.keras.layers.ConvLSTM2D(
    filters=32,
    kernel_size = (3,3),
    padding = "same",
    return_sequences = True,
    activation = "tanh",
    )(inp)
    

x = tf.keras.layers.ConvLSTM2D(
    filters=16,
    kernel_size=(3, 3),
    padding="same",
    return_sequences=True,
    activation="tanh",
)(x)

x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ConvLSTM2D(
    filters=8,
    kernel_size=(3, 3),
    padding="same",
    return_sequences=False,
    activation="tanh",
)(x)

x = tf.keras.layers.Conv2D(
     filters=4, kernel_size=(3, 3), activation="relu", padding="same"
)(x)

# Next, we will build the complete model and compile it.
lstm_model = tf.keras.models.Model(inp, x)

lstm_model.compile(
    loss='mse', optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy']
)
# Define some callbacks to improve training.
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)

# Define modifiable training hyperparameters.
epochs = 100
batch_size = 8

checkpoint_filepath = '/home/zongbangtan/conv_LSTM/LSTM-more_channels/lstm_model_80.hdf5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

# Splitting into train and validation
size = int(0.8*len(X))

#Training set
X_train = X[:size,...]
# Y = np.expand_dims(Y, axis=1)
y_train = Y[:size,...]

# X_val will be the images
X_val = X[size:,...]
# Y will be current training image + 1
y_val = Y[size:,...]

print(X_val.shape)
print(y_val.shape)
print(X_train.shape)
print(y_train.shape)
# Fit the model to the training data.

# Define modifiable training hyperparameters.
epochs = 100
batch_size = 8

checkpoint_filepath = '/home/zongbangtan/conv_LSTM/LSTM-more_channels/lstm_model_80.hdf5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

# Splitting into train and validation
size = int(0.8*len(X))

#Training set
X_train = X[:size,...]
# Y = np.expand_dims(Y, axis=1)
y_train = Y[:size,...]

# X_val will be the images
X_val = X[size:,...]
# Y will be current training image + 1
y_val = Y[size:,...]
# Fit the model to the training data.
lstm_model.fit(
    X_train,
    y_train,
    batch_size=8,
    epochs=epochs,
    verbose=2,
    validation_data = (X_val,y_val),
    callbacks=[early_stopping, reduce_lr, model_checkpoint_callback],
)


