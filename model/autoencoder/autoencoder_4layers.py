import os
import numpy as np
import tensorflow as tf
import shutil
import glob as glob
import cv2
import matplotlib.pyplot as plt
import random
from PIL import Image
from tkinter import Tcl
from sklearn.model_selection import train_test_split
train_images_list = []

#dataset
mobility = ['M=1.60', 'M=1.55', 'M=1.50', 'M=1.45', 'M=1.40', 'M=1.35','M=1.30', 'M=1.25', 'M=1.20', 'M=1.15']
layer = ['layer_1','layer_39','layer_79','layer_100','layer_69','layer_90','layer_110','layer_128','layer_50','layer_100']

for mob in mobility:
    for lay in layer:
        X_train = glob.glob(f"/home/zongbangtan/data_for_lstm/gamma=50/{mob}/sliced_by_layer/{lay}/*.jpg")
        X_train = Tcl().call('lsort', '-dict', X_train)
        X_train = X_train[20:100]
        for img in X_train:
            train_images_list.append(img)
img = cv2.imread(train_images_list[0])
dims = img.shape

train_images= []

for i,path in enumerate(train_images_list):
  image = cv2.imread(train_images_list[i],cv2.COLOR_BGR2RGB)
  img = np.array(image)
  img = img.astype("float32")/255.
  train_images.append(img)
train_images = np.array(train_images)


#dividing into train and test sets

train_data, test_data = train_test_split(train_images, train_size=0.8, test_size=0.1, random_state=42, shuffle=True)

print(test_data.shape)
tf.keras.backend.clear_session()
model = tf.keras.Sequential()

#Designing the encoder section of the model
model.add(tf.keras.layers.Conv2D(input_shape = (128,128,3), filters= 64, kernel_size = (5,5), activation='sigmoid', padding='same'))
model.add(tf.keras.layers.AveragePooling2D((2,2), padding='same'))

model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='sigmoid', padding='same'))
model.add(tf.keras.layers.AveragePooling2D((2,2), padding='same'))

model.add(tf.keras.layers.Conv2D(filters=8, kernel_size=(5,5), activation='relu', padding='same'))
model.add(tf.keras.layers.AveragePooling2D((2,2), padding='same'))

model.add(tf.keras.layers.Conv2D(filters=4, kernel_size=(5,5), activation='relu', padding='same'))
model.add(tf.keras.layers.AveragePooling2D((2,2), padding='same'))
# (32,32,8)

#Deigning the Decoder section of the model

model.add(tf.keras.layers.Conv2D(filters=4, kernel_size=(5,5), activation='relu', padding='same'))
model.add(tf.keras.layers.UpSampling2D((2,2)))

model.add(tf.keras.layers.Conv2D(filters=8, kernel_size=(5,5), activation='relu', padding='same'))
model.add(tf.keras.layers.UpSampling2D((2,2)))

model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='sigmoid', padding='same'))
model.add(tf.keras.layers.UpSampling2D((2,2)))

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(5,5),  activation='sigmoid', padding='same'))
model.add(tf.keras.layers.UpSampling2D((2,2)))

model.add(tf.keras.layers.Conv2D(filters=3, kernel_size=(3,3), activation='sigmoid', padding='same'))
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse', metrics=['accuracy'])
model.summary()
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)


checkpoint_filepath = '/home/zongbangtan/conv_LSTM/autoencoder/Reconstruction_model_4/autoencoder.hdf5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)
history = model.fit(train_data,
                    train_data,
                    validation_split=0.2,
                    epochs=1000,
                    batch_size=8,
                    verbose=2,
                    shuffle=True,
                    callbacks=[model_checkpoint_callback,early_stopping, reduce_lr])


