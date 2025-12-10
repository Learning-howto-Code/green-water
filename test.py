import os
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import re

MODEL_PATH = "model.keras"
IMAGE_FOLDER = "test/"

IMG_SIZE = (224, 224)

# Load Keras model
model = tf.keras.models.load_model(MODEL_PATH)

def numeric_key(filename):
    # Extract the first number in filename for sorting
    nums = re.findall(r'\d+', filename)
    return int(nums[0]) if nums else float('inf')

def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize(IMG_SIZE)

    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, 0)

    output = model.predict(arr, verbose=0)[0]

    pred = 1 if output[0] > 0.5 else 0
    return pred

def run_folder(folder):
    y_true = []
    y_pred = []

    label_map = {"no_water": 0, "water": 1}

    for root, _, files in os.walk(folder):
        folder_name = os.path.basename(root)
        if folder_name not in label_map:
            continue

        true_label = label_map[folder_name]

        # Sort files numerically
        image_files = sorted(
            [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))],
            key=numeric_key
        )

        for filename in image_files:
            path = os.path.join(root, filename)

            pred = predict_image(path)

            y_true.append(true_label)
            y_pred.append(pred)

            print(f"{path}: true {true_label}, predicted {pred}")

    return y_true, y_pred

if __name__ == "__main__":
    y_true, y_pred = run_folder(IMAGE_FOLDER)

    cm = confusion_matrix(y_true, y_pred)
    print("\nCONFUSION MATRIX")
    print(cm)
