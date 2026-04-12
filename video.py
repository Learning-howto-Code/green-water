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
MODEL_PATH ="/Users/jakehopkins/Downloads/if_water/if_water_4-1020260410_182006.keras"

INPUT_DIR = "/Users/jakehopkins/Downloads/tp_4-10"
MOVIE_DIR = os.path.join(INPUT_DIR, "movie")
diff_on = False
# Order goes no_water, water
DATASET_FOLDERS = {
     #"#train":  "/Users/jakehopkins/Downloads/if_water/food_clean/train",
    # "val": "/Users/jakehopkins/Downloads/if_water/food_clean/val",
    "test": INPUT_DIR
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
    if diff_on == True:
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

def run_folder(folder):
    predictions = []
    
    # Ensure movie directory exists
    os.makedirs(MOVIE_DIR, exist_ok=True)

    print(f"Processing images in: {folder}")
    
    # Sort files chronologically by timestamp embedded in filename.
    image_files = sorted(
        [f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))],
        key=lambda f: (chronological_key(f), f)
    )
    
    if len(image_files) == 0:
        print(f"WARNING: No images found in '{folder}'")
        return predictions
    
    print(f"Found {len(image_files)} images to process")
    
    for filename in image_files: 
        path = os.path.join(folder, filename)

        pred = predict_image(path)
        score = float(np.squeeze(pred))
        pred_str = np.array2string(
            pred,
            precision=6,
            floatmode="fixed",
            suppress_small=True
        )

        predictions.append({"filename": filename, "score": score, "prediction": pred_str})
        print(f"{path}: predicted {pred_str}")
        
        if score < 0.5:
            color_bgr = (0, 255, 0)   # green
        else:
            color_bgr = (0, 0, 255)   # red
        
        path_read = cv.imread(path)
        h, w = path_read.shape[:2]
        pad = 20
        size = 60
        x1, y1 = pad, h - pad - size
        x2, y2 = x1 + size, y1 + size
        cv.rectangle(path_read, (x1, y1), (x2, y2), color_bgr, thickness=-1)
        cv.imwrite(f"{MOVIE_DIR}/{filename}.png", path_read)
        
    return predictions
def evaluate_all_datasets(dataset_folders):
    results = {}
    
    for name, folder in dataset_folders.items():
        if not os.path.exists(folder):
            print(f"\nWarning: {folder} does not exist, skipping {name} dataset.")
            continue
            
        print(f"\n{'='*50}")
        print(f"Processing {name.upper()} dataset: {folder}")
        print('='*50)
        
        predictions = run_folder(folder)
        if len(predictions) > 0:
            results[name] = predictions
            print(f"\nProcessed {len(predictions)} images")
        else: 
            print(f"No images found in {folder}")
    
    return results

def generate_filelist():
    """Generate filelist.txt for FFmpeg concat demuxer"""
    if not os.path.exists(MOVIE_DIR):
        print(f"Error: {MOVIE_DIR} does not exist")
        return False
    
    # Get all PNG files and sort chronologically
    all_files = os.listdir(MOVIE_DIR)
    print(f"Files in {MOVIE_DIR}: {len(all_files)} total")
    
    png_files = sorted(
        [f for f in all_files if f.lower().endswith(".png")],
        key=lambda f: (chronological_key(f), f)
    )
    
    if not png_files:
        print(f"Error: No PNG files found in {MOVIE_DIR}")
        print(f"Total files in directory: {len(all_files)}")
        return False
    
    # Rename files to sequential numbering for image2 demuxer
    for i, png_file in enumerate(png_files):
        old_path = os.path.join(MOVIE_DIR, png_file)
        # Create padded filename: frame_0001.png, frame_0002.png, etc.
        new_filename = f"frame_{i+1:04d}.png"
        new_path = os.path.join(MOVIE_DIR, new_filename)
        os.rename(old_path, new_path)
    
    print(f"Renamed {len(png_files)} files to sequential order")
    return True

if __name__ == "__main__":

    results = evaluate_all_datasets(DATASET_FOLDERS)
    
    # Print summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print('='*50)
    for name, predictions in results.items():
        avg_score = np.mean([p["score"] for p in predictions])
        print(f"{name.upper()}: {len(predictions)} images, Average confidence: {avg_score:.4f}")
    
    # Generate filelist for FFmpeg concat
    print(f"\n{'='*50}")
    print("Creating video from images")
    print('='*50)
    if generate_filelist():
        # Use image2 demuxer with sequential filenames
        os.system(f"ffmpeg -framerate 30 -i {MOVIE_DIR}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p {INPUT_DIR}/tp_if_water.mp4")
    else:
        print("Skipping video creation - no PNG files to process")