import os
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import re

# MODEL_PATH = input("enter file path")
# MODEL_PATH= str(MODEL_PATH)
MODEL_PATH = "/Users/jakehopkins/Downloads/if_water/food_clean_noaug.keras"
# Define all dataset folders
DATASET_FOLDERS = {
     #"train":  "/Users/jakehopkins/Downloads/if_water/Clean_Dirty/train",
     "val": "/Users/jakehopkins/Downloads/if_water/food_clean/val",
    #"test": "/Users/jakehopkins/Downloads/if_water/Clean_Dirty/val"
}

IMG_SIZE = (224, 224)

# Load Keras model
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

def numeric_key(filename):
    # Extract the first number in filename for sorting
    nums = re.findall(r'\d+', filename)
    return int(nums[0]) if nums else float('inf')

def _input_mode():
    # Infer expected input channels from the model input shape
    shape = model.input_shape
    channels = shape[-1] if isinstance(shape, (list, tuple)) else 3
    if channels == 1:
        return "L"  # grayscale
    return "RGB"


def predict_image(img_path):
    img = Image.open(img_path).convert(_input_mode())
    img = img.resize(IMG_SIZE)

    arr = np.array(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)
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

        # Sort files numerically
        image_files = sorted(
            [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))],
            key=numeric_key
        )
        if len(image_files) == 0:
            print(f"WARNING: No images loaded for class '{folder_name}'")
        for filename in image_files: 
            path = os.path.join(root, filename)

            pred = predict_image(path)
            pred_idx = int(np.argmax(pred))

            pred_str = np.array2string(
                pred,
                precision=6,
                floatmode="fixed",
                suppress_small=True
            )

            y_true.append(true_label)
            y_pred.append(pred_idx)
            print(f"{path}: true {true_label}, predicted {pred_str}")
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