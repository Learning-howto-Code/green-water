# ignore errors in the imports
#SSH at (venv) Abrahams-MacBook-Pro:if_water abrahamhopkins$ 
import random
import numpy as np
from keras.layers import *
from keras.models import *
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
from tensorflow.keras.utils import Sequence, load_img, img_to_array

np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

with open("delta.json") as f:
    diff_map = json.load(f)
    diff_map = {item["filepath"]: item["diff"] for item in diff_map}

epochs = 25

# Data Generators
paths="/Users/jakehopkins/Downloads/if_water/food_clean/"
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    )
eval_datagen = ImageDataGenerator(rescale=1./255)
batch_size = 32
image_size = (224, 224)
class_mode = 'binary'


class DiffSequence(Sequence): # custom data gen
    def __init__(self, filepaths, labels, diff_map, datagen, batch_size, image_size, shuffle=True):
        self.filepaths = np.array(filepaths)
        self.labels = np.array(labels)
        self.diff_map = diff_map
        self.datagen = datagen
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.filepaths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.filepaths) / self.batch_size))

    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_paths = self.filepaths[batch_indexes]
        batch_labels = self.labels[batch_indexes]

        images = []
        for path in batch_paths:
            img = load_img(path, target_size=self.image_size)
            arr = img_to_array(img).astype("float32")
            if self.datagen is not None:
                arr = self.datagen.random_transform(arr)
                arr = self.datagen.standardize(arr)
            else:
                arr = arr / 255.0

            diff = float(self.diff_map.get(path, 0.0))
            diff_channel = np.full((*self.image_size, 1), diff, dtype=np.float32)
            arr = np.concatenate([arr, diff_channel], axis=-1)
            images.append(arr)

        return np.array(images), np.array(batch_labels)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

train_base = train_datagen.flow_from_directory(
    directory= paths + "train",
    batch_size=batch_size,
    target_size=image_size,
    class_mode=class_mode,
    color_mode='rgb',
    shuffle=False,
    seed=42
)
valid_base = eval_datagen.flow_from_directory(
    directory= paths + "val",
    batch_size=batch_size,
    target_size=image_size,
    class_mode=class_mode,
    color_mode='rgb',
    shuffle=False,
    seed=42
)
test_base = eval_datagen.flow_from_directory(
    directory= paths + "test",
    batch_size=batch_size,
    target_size=image_size,
    class_mode=class_mode,
    color_mode='rgb',
    shuffle=False,
    seed=42
)

train_data = DiffSequence(train_base.filepaths, train_base.classes, diff_map, train_datagen, batch_size, image_size, shuffle=True)
valid_data = DiffSequence(valid_base.filepaths, valid_base.classes, diff_map, eval_datagen, batch_size, image_size, shuffle=False)
test_data = DiffSequence(test_base.filepaths, test_base.classes, diff_map, eval_datagen, batch_size, image_size, shuffle=False)


model = Sequential([
layers.Input(shape=(224, 224, 4)),   # define input once (RGB + diff)
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
optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
loss='binary_crossentropy',
metrics=['accuracy']
)
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    start_from_epoch=3,
    min_delta=0.01,
    restore_best_weights=True
)

history = model.fit(
    train_data,
    verbose=1,
    validation_data=valid_data,
    epochs=epochs,
    callbacks=[early_stop]
)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model.save(f"binary_aug{timestamp}.keras")
test_loss, test_acc = model.evaluate(test_data)
print("Test accuracy:", test_acc)

#runs eval from other file to keep training script clean
from utils import plot, matrix, precision_recall
plot(history, timestamp)
matrix(model, test_data)
precision_recall(model, test_data)
