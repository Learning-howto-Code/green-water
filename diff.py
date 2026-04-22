import cv2 as cv
import numpy as np
import os
import json
import re

lookback = 1
file = "delta.json"
out_dir = "/Users/jakehopkins/Downloads/if_water/food_diff/"


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
# with open("delta.json") as f: 
#     data = json.load(f)
#     data = {item["filepath"]: item["diff"] for item in data}
#     print(data["/Users/jakehopkins/Downloads/if_water/food_clean/train/food/00662_img_20260124_130741_659.jpg"])

def diffs():
    global new_filepath, diff
    old_data = []
    if os.path.exists(file):
        with open(file, "r") as f: 
            try:
                loaded = json.load(f)
                old_data = loaded if isinstance(loaded, list) else [] # test case for empty file
            except json.JSONDecodeError:
                old_data = []

    for dirpath in os.walk("/Users/jakehopkins/Downloads/if_water/food_clean/"):
        dir = dirpath[0]
        print(dir)
        files = [f for f in os.listdir(dir) if f.endswith('.jpg')]
        files = sorted(files, key=extract_timestamp)  # sort by embedded timestamp
        #old_filepath = files[0]
        for i in range(len(files)):
            

            old_idx = max(0, i - lookback)
            old_filepath = os.path.join(dir, files[old_idx])
            new_filepath = os.path.join(dir, files[i])

            
            old = cv.imread(old_filepath)# converts filepath to array thing
            new = cv.imread(new_filepath)
            if old is  None or new is  None:
                print("had to skip")
                continue
            old = cv.resize(old, (224, 224)) 
            new = cv.resize(new, (224, 224))

        
            diff = cv.absdiff(old, new) #creates a new arrary from 2 arrays

            base = os.path.basename(new_filepath)
            base = base[:-4]  # Remove ".jpg" extension
            out_path = os.path.join(out_dir, f"{base}_diff.jpg")
            cv.imwrite(out_path, diff)

            # cv.imshow("diff", diff)
            # cv.waitKey(0)
            # cv.destroyAllWindows()


            old_filepath = new_filepath

            # print(f"new filepath {new_filepath}")
            # print(f"old filepath {old_filepath}")
            # print(f"loop number {i}")
            # print(f"difference is {diff} out 255") 

            old_data.append({
                "filepath": new_filepath,
                "diff_path": f"{out_dir}{base}_diff.jpg"
                })
    with open(file, "w") as f:
        json.dump(old_data, f, indent=2)


    
diffs()
