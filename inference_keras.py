import numpy as np
import cv2 as cv
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

img_dir = '/Users/jakehopkins/Downloads/6-6'
true_label = "water"  # set to "water" or "no water"

keras_model_path = '/Users/jakehopkins/Downloads/if_water/new_aug_if_water20260607_175647.keras'
model = tf.keras.models.load_model(keras_model_path, compile=False)
print("Model loaded:", keras_model_path)
print("Input shape:", model.input_shape)

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
image_files = [
    f for f in os.listdir(img_dir)
    if os.path.splitext(f)[1].lower() in VALID_EXTS
]

if not image_files:
    print(f"No images found in {img_dir}")
    exit(1)

y_true = []
y_pred = []

for fname in image_files:
    path = os.path.join(img_dir, fname)
    img = cv.imread(path)
    if img is None:
        print(f"Skipping (unreadable): {fname}")
        continue

    img = cv.resize(img, (224, 224))
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_array = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)

    score = float(model(img_array, training=False)[0][0])
    pred_label = "water" if score > 0.5 else "no water"
    print(f"{fname}: {score:.4f} → {pred_label}")

    y_true.append(true_label)
    y_pred.append(pred_label)

labels = ["water", "no water"]
cm = confusion_matrix(y_true, y_pred, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap="Blues")
plt.title("if_water Keras Confusion Matrix")
plt.tight_layout()
plt.show()
