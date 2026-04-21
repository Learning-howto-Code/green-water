import cv2 as cv
import numpy as np
import json

img = '/Users/jakehopkins/Downloads/if_water/poop/test/poop/, 2026-04-11-19-13-41-523.jpg'
with open("poop_delta.json") as f:
    diff_map = json.load(f)

diff_map = {item["filepath"]: item["diff_path"] for item in diff_map}

diff_img = diff_map.get(img) 
# diff_img = diff_img + "_diff.jpg"

print(diff_img) 
cv.imread(diff_img)
cv.imshow("diff", cv.imread(diff_img))
cv.waitKey(0)
cv.destroyAllWindows()
    