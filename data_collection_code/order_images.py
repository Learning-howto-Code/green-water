import os
import shutil
from datetime import datetime

SOURCE_DIR = "/Users/jakehopkins/Downloads/if_water/data/Clean_Dirty/train/food"
DEST_DIR = '/Users/jakehopkins/Downloads/if_water/data/Clean_Dirty/train/food ordered'

os.makedirs(DEST_DIR, exist_ok=True)

def extract_timestamp(filename):
    parts = filename.split("_")
    date = parts[1]
    time = parts[2]
    return datetime.strptime(date + time, "%Y%m%d%H%M%S")

files = [
    f for f in os.listdir(SOURCE_DIR)
    if f.lower().endswith(".jpg")
]

files_sorted = sorted(files, key=extract_timestamp)

for i, filename in enumerate(files_sorted):
    new_name = f"{i:05d}_{filename}"
    shutil.copy2(
        os.path.join(SOURCE_DIR, filename),
        os.path.join(DEST_DIR, new_name)
    )