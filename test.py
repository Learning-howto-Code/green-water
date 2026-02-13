import cv2 as cv
import numpy as np

img1= cv.imread("/Users/jakehopkins/Downloads/if_water/Clean_Dirty/train/clean/img_20260125_113927_1.jpg")
img2= cv.imread("/Users/jakehopkins/Downloads/if_water/Clean_Dirty/train/toilet/img_20260125_112105_7.jpg")

diff = cv.absdiff(img1, img2)
diff = np.mean(diff)

print(f"{diff} out of 255")
