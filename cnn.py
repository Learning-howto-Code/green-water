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


# Filepaths
train_paths=[ "/Users/jakehopkins/Downloads/if_water/pi_data/train"] #trying adding prod data to train

val_path= "/Users/jakehopkins/Downloads/if_water/pi_data/val"
test_path="/Users/jakehopkins/Downloads/if_water/pi_data/test"


datagen= ImageDataGenerator(rescale=1./255)
batch_size = 32
image_size = (224, 224)
class_mode = 'binary'

# Data Generators
train_datasets = []
for path in train_paths: 
    ds = tf.keras. utils.image_dataset_from_directory(
        path,
        batch_size=batch_size,
        image_size=image_size,
        color_mode='grayscale',
        label_mode='binary',
        seed=42
    )
    train_datasets.append(ds)

train_data = train_datasets[0]
for ds in train_datasets[1:]:
    train_data = train_data.concatenate(ds)
train_data = train_data.map(lambda x, y: (x / 255.0, y))


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
layers.RandomZoom(0.25),
layers.RandomTranslation(0, 0.4),
layers.RandomZoom(height_factor=(-0.4, 0.4), width_factor=(-0.2, 0.2)),#randomyl zooms, alt to cropping
layers.RandomFlip("horizontal_and_vertical"),
], name="data_augmentation")


model.compile(
optimizer='adam',
loss='binary_crossentropy',
metrics=['accuracy']
)
history = model.fit(
train_data,
verbose=0,
validation_data=valid_data,
epochs=8
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
