# ignore errors in the imports
#SSH at (venv) Abrahams-MacBook-Pro:if_water abrahamhopkins$ 
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import *
from keras.models import *
from keras.preprocessing import image
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

path_to_folder = "/Users/jakehopkins/Downloads/if_water"
# Filepaths

train_path= "//Users/abrahamhopkins/Downloads/Jakes_Model/if_water/train"
val_path= "/Users/abrahamhopkins/Downloads/Jakes_Model/if_water/val"
test_path="/Users/abrahamhopkins/Downloads/Jakes_Model/if_water/train"


datagen= ImageDataGenerator(rescale=1./255)
#prepares imgs for training
batch_size = 32
image_size = (224, 224)
class_mode = 'binary'

# Data Generators
train_data = datagen.flow_from_directory(
    train_path,
    batch_size=batch_size,
    target_size=image_size,
    class_mode=class_mode,
    color_mode='grayscale',
    seed=42
)

valid_data = datagen.flow_from_directory(
    val_path,
    batch_size=batch_size,
    target_size=image_size,
    class_mode=class_mode,
    color_mode='grayscale',
    seed=42
)
test_data = datagen.flow_from_directory(
    test_path,
    batch_size=batch_size,
    target_size=image_size,
    class_mode=class_mode,
    color_mode='grayscale',
    seed=42
)
data_augmentation = tf.keras.Sequential([
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0, 0.25),
    layers.RandomCrop(0.1, 0.25),
    layers.RandomFlip("horizontal_and_vertical"),
], name="data_augmentation")

model = Sequential([
    layers.Input(shape=(224, 224, 3)),   # define input once
    data_augmentation,
    layers.Conv2D(16, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, (3,3), activation='relu'),
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
    validation_data=valid_data,
    epochs=5
)
tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    verbose=1,
    start_from_epoch=3
)
model.save("model.keras")
test_loss, test_acc = model.evaluate(test_data)
print("Test accuracy:", test_acc)