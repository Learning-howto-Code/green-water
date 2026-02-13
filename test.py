import cv2 as cv
import numpy as np
import os

def diffs():
    dir = "/Users/jakehopkins/Downloads/if_water/food_clean/train/clean"
    files = [f for f in os.listdir(dir) if f.endswith('.jpg')] # so it doesn't get thrown off w
    old_filepath = "/Users/jakehopkins/Downloads/if_water/food_clean/test/clean/img_20260125_113927_2.jpg"
    for i in range(len(files)):
        files = [f for f in os.listdir(dir) if f.endswith('.jpg')]
        files = sorted(files)  # sorts the files by timetamp TIMESTAMPS SKIP A FEW NUMBERS
        new_filepath = os.path.join(dir, files[i])

        print(f"new filepath {new_filepath}")
        print(f"old filepath {old_filepath}")
        old = cv.imread(old_filepath)
        print(f"loop number {i}")
        new = cv.imread(new_filepath)
    
        diff = cv.absdiff(old, new)
        diff = np.average(diff)

        old_filepath = new_filepath
        # print(f"old{old} new{new}")
        print(f"difference is {diff} out 255")
diffs()
