import os
import board
import neopixel_spi as neopixel
from picamera2 import Picamera2
from time import sleep
import cv2
import datetime as datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
dir = os.path.join(ROOT, "data/2026-09-10")
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (640, 360)}, buffer_count=4)
picam2.configure(config)
picam2.start()

#sets up led ring
pixels = neopixel.NeoPixel_SPI(
    board.SPI(), 30, brightness=1.0, auto_write=False, pixel_order=neopixel.GRB
)
pixels.fill((220, 240, 120))
pixels.show()  # commit/send to LEDs

sleep(.5)
ID = 1
while True:
    img = picam2.capture_array()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #the pi cam takes in BGR
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + f"-{datetime.datetime.now().microsecond // 1000:03d}"
    img_name = f"{dir}/ID#{ID}, {timestamp}.jpg"
    x =+ 1
    cv2.imwrite(img_name,img)
    print(f"took {img_name}")
    sleep(.1)