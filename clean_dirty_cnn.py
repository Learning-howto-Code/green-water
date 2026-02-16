# ignore errors in the imports
#SSH at (venv) Abrahams-MacBook-Pro:if_water abrahamhopkins$ 
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import *
from keras.models import *
from datetime import datetime
from keras.preprocessing import image
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2 as cv
import os
import json

with open("delta.json") as f:
    data = json.load(f)
    data = {item["filepath"]: item["diff"] for item in data}
    print(data["/Users/jakehopkins/Downloads/if_water/food_clean/train/food/00662_img_20260124_130741_659.jpg"])

epochs = 3

# Data Generators
paths="/Users/jakehopkins/Downloads/if_water/food_clean/"
datagen= ImageDataGenerator(rescale=1./255)
batch_size = 32
image_size = (224, 224)
class_mode = 'binary'

train_data = datagen.flow_from_directory(
    directory= paths + "train",
    batch_size=batch_size,
    target_size=image_size,
    class_mode=class_mode,
    color_mode='rgb',
    seed=42
)
print(f"filepaths: {train_data.filenames[:10]}")
valid_data = datagen.flow_from_directory(
    directory= paths + "val",
    batch_size=batch_size,
    target_size=image_size,
    class_mode=class_mode,
    color_mode='rgb',
    seed=42
)
test_data = datagen.flow_from_directory(
    directory= paths + "test",
    batch_size=batch_size,
    target_size=image_size,
    class_mode=class_mode,
    color_mode='rgb',
    seed=42
)

model = Sequential([
layers.Input(shape=(224, 224, 3)),   # define input once
layers.Conv2D(16, (3,3), activation='relu'),
layers.Dropout(0.2),
layers.MaxPooling2D(),
layers.Conv2D(32, (3,3), activation='relu'),
layers.Dropout(0.1),
layers.MaxPooling2D(),
layers.Flatten(),
layers.Dense(64, activation='relu'),
layers.Dense(1, activation='sigmoid')
])

model.compile(
optimizer='adam',
loss='binary_crossentropy',
metrics=['accuracy']
)
history = model.fit(
train_data,
verbose=1,
validation_data=valid_data,
epochs=epochs
)
tf.keras.callbacks.EarlyStopping(
monitor='val_loss',
patience=3,
start_from_epoch=3,
min_delta=0.01,
)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model.save(f"testing{timestamp}.keras")
test_loss, test_acc = model.evaluate(test_data)
print("Test accuracy:", test_acc)

#runs eval from other file to keep training script clean
from model_eval import plot, matrix
plot(history, timestamp)
matrix(model, test_data)