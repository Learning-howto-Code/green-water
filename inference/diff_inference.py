import numpy as np
import cv2 as cv
import tensorflow as tf
import os
import re
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

img_dir ='/Users/jakehopkins/Downloads/9-water'
true_label = "water"  # set to "water" or "no water"
keras_model_path = '/Users/jakehopkins/Downloads/if_water/water_more_data20260912_145101.keras'
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


def collect_groups(root):
    """One sorted file list per directory, so diffs never pair across dirs.

    A flat directory yields a single group; nested ones yield a group per
    leaf. Mirrors how diff.py walks the tree.
    """
    groups = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()  # deterministic descent
        files = [f for f in filenames
                 if os.path.splitext(f)[1].lower() in VALID_EXTS]
        if files:
            groups.append((dirpath, sorted(files, key=extract_timestamp)))
    return sorted(groups)


groups = collect_groups(img_dir)

if not groups:
    print(f"No images found in {img_dir}")
    exit(1)

total = sum(len(f) for _, f in groups)
print(f"Found {total} images in {len(groups)} director{'y' if len(groups) == 1 else 'ies'}")

_cache = {}


def load_resized(path):
    """Decoded BGR frame at model resolution, or None if unreadable."""
    if path not in _cache:
        img = cv.imread(path)
        _cache[path] = None if img is None else cv.resize(img, image_size)
    return _cache[path]


def make_diff(dirpath, files, idx):
    """absdiff against the frame `lookback` back within this dir, as diff.py does."""
    new = load_resized(os.path.join(dirpath, files[idx]))
    old = load_resized(os.path.join(dirpath, files[max(0, idx - lookback)]))
    if new is None or old is None:
        return None, 0.0
    diff = cv.absdiff(old, new)
    if match_jpeg_roundtrip:
        ok, buf = cv.imencode(".jpg", diff)
        if ok:
            diff = cv.imdecode(buf, cv.IMREAD_COLOR)
    mini_diff = float(np.average(diff))
    return cv.cvtColor(diff, cv.COLOR_BGR2RGB), mini_diff


y_true = []
y_pred = []
zero_filled = 0

for dirpath, files in groups:
    if len(groups) > 1:
        print(f"\n--- {os.path.relpath(dirpath, img_dir)} ({len(files)} images)")
    # cache only needs the previous frame of the current dir
    _cache.clear()

    for i, fname in enumerate(files):
        path = os.path.join(dirpath, fname)
        img = load_resized(path)
        if img is None:
            print(f"Skipping (unreadable): {fname}")
            continue

        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        arr = np.array(img, dtype=np.float32) / 255.0

        if diff_on:
            diff, mini_diff = make_diff(dirpath, files, i)
            if diff is None:
                # training fell back to zeros here too
                diff_arr = np.zeros((*image_size, 3), dtype=np.float32)
                zero_filled += 1
            else:
                diff_arr = np.array(diff, dtype=np.float32) / 255.0
            arr = np.concatenate([arr, diff_arr], axis=-1)
        else:
            mini_diff = None

        img_array = np.expand_dims(arr, axis=0)

        score = float(model(img_array, training=False)[0][0])
        pred_label = "water" if score > 0.5 else "no water"
        suffix = f"  diff={mini_diff:.4f}" if mini_diff is not None else ""
        print(f"{fname}: {score:.4f} → {pred_label}{suffix}")

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
