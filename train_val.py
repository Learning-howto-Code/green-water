import os
import random
import shutil

# Set your paths here
base_dir = "/Users/jakehopkins/Downloads/if_water/temp"
train_dir = "/Users/jakehopkins/Downloads/if_water/poop/train/clean"
val_dir   = "/Users/jakehopkins/Downloads/if_water/poop/val/clean"
test_dir = "/Users/jakehopkins/Downloads/if_water/poop/test/clean"
val_ratio = 0.15
test_ratio = 0.25
train_ratio = 0.6
# Allowed image extensions
IMG_EXT = (".jpg", ".jpeg", ".png")

def move_split(base_dir, train_dir, val_dir, test_dir, val_ratio=val_ratio, test_ratio=test_ratio, train_ratio=train_ratio):
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-9:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    # List only image files
    files = [f for f in os.listdir(base_dir)
             if f.lower().endswith(IMG_EXT)]

    if not files:
        print("No image files found in base directory.")
        return

    # Shuffle list to randomize selection
    random.shuffle(files)

    # Build non-overlapping splits from one shuffled list
    total = len(files)
    tr_val = int(total * train_ratio)
    v_val = int(total * val_ratio)

    train_files_to_move = files[:tr_val]
    val_files_to_move = files[tr_val:tr_val + v_val]
    test_files_to_move = files[tr_val + v_val:]


    for fname in val_files_to_move:
        src = os.path.join(base_dir, fname)
        dst = os.path.join(val_dir, fname)
        shutil.move(src, dst)
    for fname in test_files_to_move:
        src = os.path.join(base_dir, fname)
        dst = os.path.join(test_dir, fname)
        shutil.move(src, dst)
    for fname in train_files_to_move:
        src = os.path.join(base_dir, fname)
        dst = os.path.join(train_dir, fname)
        shutil.move(src, dst)

    print("Done. No contamination; files were moved, not copied.")
    print(f"Source remaining: {len([f for f in os.listdir(base_dir) if f.lower().endswith(IMG_EXT)])}")
    print(f"Val count:       {len(os.listdir(val_dir))}")
    print(f"Test count:      {len(os.listdir(test_dir))}")
    print(f"Train count:      {len(os.listdir(train_dir))}")


move_split(base_dir, train_dir, val_dir, test_dir)

