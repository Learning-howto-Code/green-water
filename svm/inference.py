import os
import json
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

MODEL_PATH = "/Users/jakehopkins/Downloads/if_water/food_full_diff20260425_162603.keras"
IMAGE_FOLDER = "/Users/jakehopkins/Downloads/if_water/food_clean/train/clean"
IMG_SIZE = (224, 224)
diff_on = True

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
feature_extractor = tf.keras.Model(
    inputs=model.inputs,
    outputs=model.layers[-2].output
)

batch = []
old_arr = None
for filename in sorted(os.listdir(IMAGE_FOLDER)):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    img_path = os.path.join(IMAGE_FOLDER, filename)
    img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0  # (224, 224, 3)
    if diff_on == True:
        diff = cv2.absdiff(old_arr, arr) if old_arr is not None else np.zeros_like(arr)
        old_arr = arr

        combined = np.concatenate([arr, diff], axis=-1)  # (224, 224, 6)
        batch.append(combined)
batch = np.array(batch)
features = feature_extractor.predict(batch, verbose=1).tolist()
labels = [1] * len(features)  # all Dirty

with open('normal_data.json', 'w') as f:
    json.dump({"features": features, "labels": labels}, f, indent=2)

print(f"Saved {len(features)} samples to normal_data.json")
