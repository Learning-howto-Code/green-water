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
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore


# Filepaths
train_paths="/Users/jakehopkins/Downloads/if_water/data/if_water_data/train"
val_path= "/Users/jakehopkins/Downloads/if_water/data/if_water_data/val"
test_path="/Users/jakehopkins/Downloads/if_water/data/if_water_data/test"
# Order goes no_water, water

train_datagen= ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.2,
    # shear_range=0.2,
    # zoom_range=0.2,
    # horizontal_flip=True,
    )
eval_datagen = ImageDataGenerator(rescale=1./255)
batch_size = 32
image_size = (224, 224)
class_mode = 'binary'

# Data Generators
train_data = train_datagen.flow_from_directory(
    train_paths,
    batch_size=batch_size,
    target_size=image_size,
    class_mode=class_mode,
    color_mode='rgb',
    seed=42,
    shuffle=True
)
val_data = eval_datagen.flow_from_directory(
    val_path,
    batch_size=batch_size,
    target_size=image_size,
    class_mode=class_mode,
    color_mode='rgb',
    seed=42,
    shuffle=False
)
test_data = eval_datagen.flow_from_directory(
    test_path,
    batch_size=batch_size,
    target_size=image_size,
    class_mode=class_mode,
    color_mode='rgb',
    seed=42,
    shuffle=False,
)
import numpy as np
print("Train class distribution:", dict(zip(*np.unique(train_data.classes, return_counts=True))))
print("Class indices:", train_data.class_indices)

model = Sequential([
layers.Input(shape=(224, 224, 3)),
layers.Conv2D(32, (3,3), activation='relu'),
layers.BatchNormalization(),
layers.MaxPooling2D(),
layers.Conv2D(64, (3,3), activation='relu'),
layers.BatchNormalization(),
layers.MaxPooling2D(),
layers.Conv2D(128, (3,3), activation='relu'),
layers.BatchNormalization(),
layers.MaxPooling2D(),
layers.GlobalAveragePooling2D(),  # replaces Flatten — kills the 100k param explosion
layers.Dense(128, activation='relu'),
layers.Dropout(0.4),
layers.Dense(1, activation='sigmoid')
])

model.compile(
optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # default is 1e-3
loss='binary_crossentropy',
metrics=['accuracy']
)
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    start_from_epoch=3,
    min_delta=0.01,
)
history = model.fit(
    train_data,
    verbose=1,
    validation_data=val_data,
    epochs=15,
    callbacks=[early_stop],
)

# plot
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.title('Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title('Loss')
plt.legend()
plt.tight_layout()
plt.show()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model.save(f"no_BN_if_water{timestamp}.keras")
test_loss, test_acc = model.evaluate(test_data)
print("Test accuracy:", test_acc)

y_pred = (model.predict(test_data) > 0.5).astype(int).flatten()
y_true = test_data.classes
class_names = list(test_data.class_indices.keys())  # ['no_water', 'water']

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues")
plt.title("if_water Confusion Matrix")
plt.tight_layout()
plt.show()