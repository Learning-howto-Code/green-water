import os
import random
import shutil

# Set your paths here
train_dir = "/Users/abrahamhopkins/Downloads/Jakes_Model/if_water/pi_data/train/water"
val_dir   = "/Users/abrahamhopkins/Downloads/Jakes_Model/if_water/pi_data/val/water"

# Allowed image extensions
IMG_EXT = (".jpg", ".jpeg", ".png")

def move_split(train_dir, val_dir, val_ratio=0.15):
    os.makedirs(val_dir, exist_ok=True)

    # List only image files
    files = [f for f in os.listdir(train_dir)
             if f.lower().endswith(IMG_EXT)]

    if not files:
        print("No image files found in train directory.")
        return

    # Shuffle list to randomize selection
    random.shuffle(files)

    # Number of files to move
    n_val = int(len(files) * val_ratio)
    files_to_move = files[:n_val]

    print(f"Moving {n_val} files from train → val...")

    for fname in files_to_move:
        src = os.path.join(train_dir, fname)
        dst = os.path.join(val_dir, fname)
        shutil.move(src, dst)

    print("Done. No contamination; files were moved, not copied.")
    print(f"Train remaining: {len(os.listdir(train_dir))}")
    print(f"Val count:       {len(os.listdir(val_dir))}")

move_split(train_dir, val_dir)

