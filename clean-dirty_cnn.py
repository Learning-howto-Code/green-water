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
from sklearn.metrics import confusion_matrix, classification_report


# Filepaths
train_paths="/Users/jakehopkins/Downloads/if_water/Clean_Dirty/train"
val_path= "/Users/jakehopkins/Downloads/if_water/Clean_Dirty/val"
test_path="/Users/jakehopkins/Downloads/if_water/Clean_Dirty/test"


img_size = (224, 224)
batch_size = 32
image_size = (224, 224)
class_mode = 'categorical'

# Data loader
datagen = ImageDataGenerator(rescale=1.0/255)

# Data Generators
data_aug = ImageDataGenerator( #data aug generator
    rotation_range=5,
    width_shift_range=0.1, 
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
)
train_data = data_aug.flow_from_directory(  #takes data_aug gen as input, instead of standard Image data gen
    train_paths,
    batch_size=batch_size,
    target_size=image_size,
    class_mode=class_mode,
    color_mode='rgb',
    seed=42
)
valid_data = datagen.flow_from_directory(
val_path,
batch_size=batch_size,
target_size=image_size,
class_mode=class_mode,
color_mode='rgb',
seed=42
)
test_data = datagen.flow_from_directory(
    test_path,
    batch_size=batch_size,
    target_size=image_size,
    class_mode=class_mode,
    color_mode='rgb',
    seed=42,
    shuffle=False
)

model = Sequential([
layers.Input(shape=(224, 224, 3)),   # define input once
layers.Conv2D(16, (3,3), activation='relu'),
layers.MaxPooling2D(),
layers.Conv2D(32, (3,3), activation='relu'),
layers.MaxPooling2D(),
layers.Flatten(),
layers.Dense(64, activation='relu'),
layers.Dense(4, activation='softmax')  
])

model.compile(
optimizer='adam',
loss='categorical_crossentropy',
metrics=['accuracy']
)
history = model.fit(
train_data,
verbose=1,
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

model.save(f"clean_dirty_data_aug{timestamp}.keras")

def plot(history, timestamp):
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='train')
    plt.plot(val_acc, label='val')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='train')
    plt.plot(val_loss, label='val')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()
    plt.close()
plot(history, timestamp)
test_loss, test_acc = model.evaluate(test_data)
print("Test accuracy:", test_acc)

# Confusion matrix on test set
def matrix():
    y_true = test_data.classes
    y_pred_probs = model.predict(test_data, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    cm = confusion_matrix(y_true, y_pred)
    print("\nCONFUSION MATRIX")
    print(cm)

    # Align target names to the labels actually present
    inv_class_map = {v: k for k, v in test_data.class_indices.items()}
    labels_present = sorted(np.unique(np.concatenate([y_true, y_pred])))
    target_names = [inv_class_map[i] for i in labels_present]

    print("\nCLASSIFICATION REPORT")
    print(classification_report(y_true, y_pred, labels=labels_present, target_names=target_names, digits=4))
matrix()