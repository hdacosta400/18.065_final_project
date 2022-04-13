from keras import layers
from keras import models
import keras
from keras.preprocessing.image import ImageDataGenerator
import os
import cv2
import numpy as np

input_shape = (144, 216)
num_classes = 2
batch_size = 20
epochs = 15

train_dir = "./train_skeleton_images/"
test_dir = "./test_skeleton_images/"
#train_skeleton_images/good/  : good images
#train_skeleton_images/bad/  : bad images
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory( train_dir, target_size=input_shape, batch_size=batch_size)
test_generator = test_datagen.flow_from_directory( test_dir, target_size=input_shape, batch_size=batch_size)

new_shape = (144, 216, 3)
model = keras.Sequential([
        keras.Input(shape=new_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history = model.fit_generator( train_generator, steps_per_epoch=100, epochs=30) 
_, acc = model.evaluate(test_generator)
print("accuracy:", acc*100)

model.save('skeleton_model')
