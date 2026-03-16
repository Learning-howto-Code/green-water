from flask import Flask, flash, render_template, render_template_string, request, Response
# from picamera2 import Picamera2
import cv2 as cv
import os
import time
import datetime

done = False
folder_time = time.strftime("%m%d%H:%M")
base_dir="/Users/jakehopkins/Downloads/if_water/data/sink"
out_dir = os.path.join(base_dir, folder_time)
os.makedirs(out_dir, exist_ok=True)
img = "/Users/jakehopkins/Downloads/if_water/food_clean_extra_clean/01442_img_20260124_130809_1446.jpg"
timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + f"-{datetime.datetime.now().microsecond // 1000:03d}"
img_name = f"{timestamp}.jpg"
cv.imwrite(os.path.join(out_dir, img_name), img)
done = True