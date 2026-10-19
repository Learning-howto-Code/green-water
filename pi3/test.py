import zmq
import time
import gpiozero
import board
import neopixel_spi as neopixel
from picamera2 import Picamera2
import cv2 as cv
from datetime import datetime

spi = board.SPI()
pixels = neopixel.NeoPixel_SPI(
    spi, 24, brightness=0.2, auto_write=False, pixel_order=neopixel.RGB
)
pixels.fill((220, 240, 120))
pixels.brightness = 0.2
pixels.show()
print("light")
#sets up camera
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (640, 360)}, buffer_count=4)
picam2.configure(config)
picam2.start()
time.sleep(2)
x=1
def take_img():
    global x
    img = picam2.capture_array()
    cv.imwrite(f"data/{x}.jpg", img)
    x+=1
time.sleep(3)
for i in range(100):
    take_img()
    time.sleep(1)
pixels.fill((0, 0, 0))
pixels.show()
pixels.deinit()