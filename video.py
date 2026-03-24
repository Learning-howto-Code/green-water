from hmac import new
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import re
import json
import cv2 as cv
from datetime import datetime


# MODEL_PATH = input("enter file path")
# MODEL_PATH= str(MODEL_PATH)
MODEL_PATH ="/Users/jakehopkins/Downloads/if_water/most_recent_model.keras"

DATASET_FOLDERS = {
     #"#train":  "/Users/jakehopkins/Downloads/if_water/food_clean/train",
    # "val": "/Users/jakehopkins/Downloads/if_water/food_clean/val",
    "test": "/Users/jakehopkins/Downloads/if_water/food_clean/test"
}

IMG_SIZE = (224, 224)

# Load Keras model
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

def chronological_key(filename):
    # Supports both patterns:
    # 1) img_20260125_114009_1247.jpg
    # 2) ID#1, 2026-03-19-17-59-23-334.jpg
    m1 = re.search(r"(\d{8})_(\d{6})_(\d+)", filename)
    if m1:
        date_str, time_str, ms_str = m1.groups()
        dt = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")
        # Normalize to microseconds (max 6 digits).
        us = int(ms_str[:6].ljust(6, "0"))
        return dt.replace(microsecond=us)

    m2 = re.search(r"(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d+)", filename)
    if m2:
        y, mo, d, h, mi, s, ms_str = m2.groups()
        us = int(ms_str[:6].ljust(6, "0"))
        return datetime(int(y), int(mo), int(d), int(h), int(mi), int(s), us)

    # Keep unknown filenames at the end, then sort by name for deterministic order.
    return datetime.max
# def numeric_key(filename):
#     # Extract the first number in filename for sorting
#     nums = re.findall(r'\d+', filename)
#     return int(nums[0]) if nums else float('inf')

def _input_mode():
    # Infer expected input channels from the model input shape
    shape = model.input_shape
    channels = shape[-1] if isinstance(shape, (list, tuple)) else 3
    if channels == 1:
        return "L"  # grayscale
    return "RGB"

old_file = None

def predict_image(img_path):
    global old_file
    img = Image.open(img_path).convert(_input_mode())
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)
    
    # Add diff as 4th channel
    new_file = img
    new_file_read = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)

    
    if old_file is None or old_file.shape != new_file_read.shape:
        diff = 0
    else:
        diff = cv.absdiff(old_file, new_file_read)
        diff = np.average(diff)
    old_file = new_file_read.copy()


    diff_channel = np.full((IMG_SIZE[0], IMG_SIZE[1]), diff, dtype=np.float32)
    print(f"diff is {diff}")
    arr = np.dstack((arr, diff_channel))
    
    arr = np.expand_dims(arr, axis=0)

    output = model.predict(arr, verbose=0)[0]
    return output

def _get_label_map(root_folder):
    class_names = sorted(
        [
            d
            for d in os.listdir(root_folder)
            if os.path.isdir(os.path.join(root_folder, d))
        ]
    )
    return {name: idx for idx, name in enumerate(class_names)}, class_names


def run_folder(folder, label_map):
    y_true = []
    y_pred = []

    for root, _, files in os. walk(folder):
        print("Walking:", root)
        folder_name = os.path.basename(root)
        if folder_name not in label_map: 
            continue

        true_label = label_map[folder_name]

        # Sort files chronologically by timestamp embedded in filename.
        image_files = sorted(
            [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))],
            key=lambda f: (chronological_key(f), f)
        )
        if len(image_files) == 0:
            print(f"WARNING: No images loaded for class '{folder_name}'")
        for filename in image_files: 
            path = os.path.join(root, filename)

            pred = predict_image(path)
            score = float(np.squeeze(pred))
            pred_idx = 1 if score > 0.5 else 0
            pred_str = np.array2string(
                pred,
                precision=6,
                floatmode="fixed",
                suppress_small=True
            )

            y_true.append(true_label)
            y_pred.append(pred_idx)
            print(f"{path}: true {true_label}, predicted {pred_str}")
            if score < 0.5:
                color_bgr = (0, 255, 0)   # green
            else:
                color_bgr = (0, 0, 255)   # red
            path_read = cv.imread (path)
            h, w = path_read.shape[:2]
            pad = 20
            size = 60
            x1, y1 = pad, h - pad - size
            x2, y2 = x1 + size, y1 + size
            cv.rectangle(path_read, (x1, y1), (x2, y2), color_bgr, thickness=-1)
            cv.imshow("Prediction", path_read)
            cv.waitKey(500)  # Display each image for 500 ms
            cv.imwrite(f"/Users/jakehopkins/Downloads/if_water/movie/{filename}.png", path_read)
            
    return y_true, y_pred
def evaluate_all_datasets(dataset_folders):
    results = {}
    
    label_map, class_names = _get_label_map(next(iter(dataset_folders.values())))

    for name, folder in dataset_folders.items():
        if not os.path.exists(folder):
            print(f"\nWarning: {folder} does not exist, skipping {name} dataset.")
            continue
            
        print(f"\n{'='*50}")
        print(f"Evaluating {name. upper()} dataset: {folder}")
        print('='*50)
        
        y_true, y_pred = run_folder(folder, label_map)
        if len(y_true) > 0:
            cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
            results[name] = {
                "confusion_matrix":  cm,
                "y_true": y_true,
                "y_pred": y_pred
            }
            
            print(f"\n{name. upper()} CONFUSION MATRIX")
            print(cm)
            print(f"Class order: {class_names}")
            
            # Calculate and display accuracy
            accuracy = np.sum(np.diag(cm)) / np.sum(cm)
            print(f"Accuracy: {accuracy:.4f}")
        else: 
            print(f"No images found in {folder}")
    
    return results

if __name__ == "__main__":

    results = evaluate_all_datasets(DATASET_FOLDERS)
    
    # Print summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print('='*50)
    for name, data in results.items():
        cm = data["confusion_matrix"]
        accuracy = np. sum(np.diag(cm)) / np.sum(cm)
        total_samples = np.sum(cm)
        print(f"{name.upper()}: {total_samples} samples, Accuracy: {accuracy:.4f}")

os.system("ffmpeg -f concat -safe 0 -i filelist.txt -r 30 -c:v libx264 -pix_fmt yuv420p video.mp4")