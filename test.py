import os
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import re

MODEL_PATH = "model_with_prod_lots_of_aug20251220_140845.keras"

# Define all dataset folders
DATASET_FOLDERS = {
    # "train":  "train/",
    # "val": "val/",
    "test": "test/"
}

IMG_SIZE = (224, 224)

# Load Keras model
model = tf. keras.models.load_model(MODEL_PATH)

def numeric_key(filename):
    # Extract the first number in filename for sorting
    nums = re.findall(r'\d+', filename)
    return int(nums[0]) if nums else float('inf')

def predict_image(img_path):
    img = Image.open(img_path).convert("L")  # Convert to grayscale (1 channel)
    img = img.resize(IMG_SIZE)

    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=-1)  # Add channel dimension:  (224, 224) -> (224, 224, 1)
    arr = np.expand_dims(arr, axis=0)   # Add batch dimension:  (224, 224, 1) -> (1, 224, 224, 1)

    output = model. predict(arr, verbose=0)[0]

    pred = 1 if output[0] > 0.5 else 0
    return pred

def run_folder(folder):
    y_true = []
    y_pred = []

    label_map = {"no_water": 0, "water": 1}

    for root, _, files in os. walk(folder):
        folder_name = os.path.basename(root)
        if folder_name not in label_map: 
            continue

        true_label = label_map[folder_name]

        # Sort files numerically
        image_files = sorted(
            [f for f in files if f.lower().endswith((".png", ".jpg", ". jpeg"))],
            key=numeric_key
        )

        for filename in image_files: 
            path = os.path.join(root, filename)

            pred = predict_image(path)

            y_true. append(true_label)
            y_pred.append(pred)

            print(f"{path}: true {true_label}, predicted {pred}")

    return y_true, y_pred

def evaluate_all_datasets(dataset_folders):
    results = {}
    
    for name, folder in dataset_folders.items():
        if not os.path.exists(folder):
            print(f"\nWarning: {folder} does not exist, skipping {name} dataset.")
            continue
            
        print(f"\n{'='*50}")
        print(f"Evaluating {name. upper()} dataset: {folder}")
        print('='*50)
        
        y_true, y_pred = run_folder(folder)
        
        if len(y_true) > 0:
            cm = confusion_matrix(y_true, y_pred)
            results[name] = {
                "confusion_matrix":  cm,
                "y_true": y_true,
                "y_pred":  y_pred
            }
            
            print(f"\n{name. upper()} CONFUSION MATRIX")
            print(cm)
            
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
        print(f"{name.upper()}: {total_samples} samples, Accuracy: {accuracy:. 4f}")
