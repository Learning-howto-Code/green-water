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

epochs = 3
# Filepaths
paths="/Users/jakehopkins/Downloads/if_water/food_clean/"

datagen= ImageDataGenerator(rescale=1./255)
batch_size = 32
image_size = (224, 224)
class_mode = 'binary'

data_aug = ImageDataGenerator( #data aug generator
    rescale=1./255,
    rotation_range=5,
    width_shift_range=0.1, 
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
)

train_data = tf.keras.utils.image_dataset_from_directory(
    directory = paths + "train/",
    batch_size=batch_size,
    image_size=image_size,
    label_mode=class_mode,
    color_mode='rgb',
    shuffle=True,
    seed=42
)
valid_data = tf.keras.utils.image_dataset_from_directory(
    directory = paths + "val/",
    batch_size=batch_size,
    image_size=image_size,
    label_mode=class_mode,
    color_mode='rgb',
    shuffle=True,
    seed=42
)
test_data = tf.keras.utils.image_dataset_from_directory(
    directory = paths +"test/",
    batch_size=batch_size,
    image_size=image_size,
    label_mode=class_mode,
    color_mode='rgb',
    shuffle=True,
    seed=42
)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_data.cache().prefetch(AUTOTUNE)
val_ds   = valid_data.cache().prefetch(AUTOTUNE)
test_ds  = test_data.cache().prefetch(AUTOTUNE)
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