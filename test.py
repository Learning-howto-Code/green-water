from flask import Flask, flash, render_template, render_template_string, request, Response
# from picamera2 import Picamera2
import cv2 as cv
import os
import time
import datetime
from pi5neo import Pi5Neo
from picamera2 import Picamera2

picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (640, 360)}, buffer_count=4)
picam2.configure(config)
picam2.start()

SPI_DEVICE = '/dev/spidev0.0' # Rpi protocol to get the timing right for the GPIOs
SPI_SPEED_KHZ = 800 #speed of SPI protocol

neo = Pi5Neo(SPI_DEVICE, 30, SPI_SPEED_KHZ)

neo.fill_strip(220, 240, 120)
neo.update_strip()  # commit/send to LEDs
time.sleep(0.5)

done = False
folder_time = time.strftime("%m:%d_%H:%M")
base_dir="/home/jake/Downloads/if-water-cnn/data/sink"
out_dir = os.path.join(base_dir, folder_time)
os.makedirs(out_dir, exist_ok=True)
frame = picam2.capture_array()
img = cv.cvtColor(frame, cv.COLOR_BGR2RGB) #the pi cam takes in BGR
timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + f"-{datetime.datetime.now().microsecond // 1000:03d}"
img_name = f"{timestamp}.jpg"
cv.imwrite(os.path.join(out_dir, img_name), img)
done = True