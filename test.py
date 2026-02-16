import cv2 as cv
import numpy as np
import os
import json

with open("delta.json") as f:
    data = json.load(f)
    data = {item["filepath"]: item["diff"] for item in data}
    print(data["/Users/jakehopkins/Downloads/if_water/food_clean/train/food/00662_img_20260124_130741_659.jpg"])

def diffs():
    global new_filepath, diff
    old_data = []
    if os.path.exists("delta.json"):
        with open("delta.json", "r") as f:
            try:
                loaded = json.load(f)
                old_data = loaded if isinstance(loaded, list) else [] # test case for empty file
            except json.JSONDecodeError:
                old_data = []
    for dirpath in os.walk("/Users/jakehopkins/Downloads/if_water/food_clean"):
        dir = dirpath[0]
        print(dir)
        files = [f for f in os.listdir(dir) if f.endswith('.jpg')]
        old_filepath = "/Users/jakehopkins/Downloads/if_water/food_clean/test/clean/img_20260125_113927_2.jpg"
        for i in range(len(files)):
            files = [f for f in os.listdir(dir) if f.endswith('.jpg')]
            files = sorted(files)  # sorts the files by timetamp TIMESTAMPS SKIP A FEW NUMBERS
            new_filepath = os.path.join(dir, files[i])

            
            old = cv.imread(old_filepath)
            
            new = cv.imread(new_filepath)
        
            diff = cv.absdiff(old, new)
            diff = np.average(diff)

            old_filepath = new_filepath
            print(f"new filepath {new_filepath}")
            print(f"old filepath {old_filepath}")
            print(f"loop number {i}")
            print(f"difference is {diff} out 255")            
            old_data.append({
                "filepath": new_filepath,
                "diff": float(diff)
            })
    with open("delta.json", "w") as f:
        json.dump(old_data, f, indent=2)
# diffs()
