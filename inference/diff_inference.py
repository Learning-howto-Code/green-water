import numpy as np
import cv2 as cv
import tensorflow as tf
import os
import re
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

img_dir = '/Users/jakehopkins/Downloads/if_water/data/if_water_data/test/no_water'
true_label = "no water"  # set to "water" or "no water"

keras_model_path = '/Users/jakehopkins/Downloads/if_water/if_water_testing.keras'
model = tf.keras.models.load_model(keras_model_path, compile=False)
print("Model loaded:", keras_model_path)
print("Input shape:", model.input_shape)

image_size = (224, 224)
lookback = 1  # must match diff.py

# 6 channels -> model wants RGB + diff, 3 -> RGB only
channels = model.input_shape[-1]
diff_on = channels == 6
if channels not in (3, 6):
    raise ValueError(f"model expects {channels} channels, only 3 or 6 supported")
print("Diff channels:", "on" if diff_on else "off")

# training diffs were written to disk as JPEG then reloaded, so they carry
# compression artifacts. Round-trip in memory to feed the model the same thing.
match_jpeg_roundtrip = True

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
image_files = [
    f for f in os.listdir(img_dir)
    if os.path.splitext(f)[1].lower() in VALID_EXTS
]

if not image_files:
    print(f"No images found in {img_dir}")
    exit(1)


def extract_timestamp(filename):
    # ID#1, 2026-03-07-08-16-56-867.jpg
    m = re.match(r'ID#\d+,\s*(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d+)', filename)
    if m:
        return tuple(int(g) for g in m.groups())
    # [NNNNN_]img_YYYYMMDD_HHMMSS_NNNN.jpg
    m = re.search(r'img_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})_(\d+)', filename)
    if m:
        return tuple(int(g) for g in m.groups())
    return (0, 0, 0, 0, 0, 0, 0)


# same ordering diff.py used to pick frame pairs
image_files = sorted(image_files, key=extract_timestamp)

_cache = {}


def load_resized(fname):
    """Decoded BGR frame at model resolution, or None if unreadable."""
    if fname not in _cache:
        img = cv.imread(os.path.join(img_dir, fname))
        _cache[fname] = None if img is None else cv.resize(img, image_size)
    return _cache[fname]


def make_diff(idx):
    """absdiff against the frame `lookback` back, mirroring diff.py."""
    new = load_resized(image_files[idx])
    old = load_resized(image_files[max(0, idx - lookback)])
    if new is None or old is None:
        return None
    diff = cv.absdiff(old, new)
    if match_jpeg_roundtrip:
        ok, buf = cv.imencode(".jpg", diff)
        if ok:
            diff = cv.imdecode(buf, cv.IMREAD_COLOR)
    mini_diff = np.average(diff)
    return cv.cvtColor(diff, cv.COLOR_BGR2RGB), mini_diff


y_true = []
y_pred = []
zero_filled = 0

for i, fname in enumerate(image_files):
    img = load_resized(fname)
    if img is None:
        print(f"Skipping (unreadable): {fname}")
        continue

    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    arr = np.array(img, dtype=np.float32) / 255.0

    if diff_on:
        diff, mini_diff = make_diff(i)
        if diff is None:
            # training fell back to zeros here too
            diff_arr = np.zeros((*image_size, 3), dtype=np.float32)
            zero_filled += 1
        else:
            diff_arr = np.array(diff, dtype=np.float32) / 255.0
        arr = np.concatenate([arr, diff_arr], axis=-1)

    img_array = np.expand_dims(arr, axis=0)

    score = float(model(img_array, training=False)[0][0])
    pred_label = "water" if score > 0.5 else "no water"
    print(f"{fname}: {score:.4f} → {pred_label}")
    print(mini_diff)

    y_true.append(true_label)
    y_pred.append(pred_label)

if zero_filled:
    print(f"WARNING: {zero_filled} frames got a zero diff (no usable previous frame)")

labels = ["water", "no water"]
cm = confusion_matrix(y_true, y_pred, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap="Blues")
plt.title("if_water Keras Confusion Matrix")
plt.tight_layout()
plt.show()
