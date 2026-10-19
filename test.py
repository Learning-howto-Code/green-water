# open ranges from 135-145
#closed ranges from 235-238
import cv2 as cv
import numpy as np
import time

open_gate = cv.imread("/Users/jakehopkins/Downloads/test-2.jpg")
closed=cv.imread("/Users/jakehopkins/Downloads/data/closed/113.jpg")

open_gate = cv.cvtColor(open_gate, cv.COLOR_RGB2GRAY)
open_gate = cv.rotate(open_gate, cv.ROTATE_180)

closed = cv.cvtColor(closed, cv.COLOR_RGB2GRAY)
closed = cv.rotate(closed, cv.ROTATE_180)

crop_open = open_gate[0:55, 300:375]
crop_closed = closed[0:55, 300:375]

closed_avg=cv.mean(crop_closed)
open_avg=cv.mean(crop_open)

def avg(img):
    if img is not None:
        img=cv.imread(img)
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        img = cv.rotate(img, cv.ROTATE_180)
        img = img[0:55, 300:375]
        return cv.mean(img)[0]
    else:
        return 0

mean=0   
for i in range(101,200):
    img = f"/Users/jakehopkins/Downloads/data/closed/{i}.jpg"
    img= avg(img)
    print(img)
    mean += img
print(mean/100)
print(f"{img} : {avg(img)}")
# print("closed"+str(closed_avg))
# print("open"+str(open_avg))
# # cv.imshow("rotate", img)
# cv.imshow("crop", open_gate[0:55, 300:375])# crops for valve gate
# cv.imshow("crop closed", closed[0:55, 300:375])# crops for valve gate

# cv.waitKey(0)          # wait for any key
# cv.destroyAllWindows()

