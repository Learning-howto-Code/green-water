import os
import pi5neo
from picamera2 import Picamera2
from time import sleep
import cv2
import datetime as datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
dir = os.path.join(ROOT, "data/4-27")
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (640, 360)}, buffer_count=4)
picam2.configure(config)
picam2.start()

#sets up led ring
SPI_DEVICE = '/dev/spidev0.0' # Rpi protocol to get the timing right for the GPIOs
SPI_SPEED_KHZ = 800 #speed of SPI protocol

neo = pi5neo.Pi5Neo(SPI_DEVICE, 30, SPI_SPEED_KHZ)
# Fill the strip with white (R,G,B = 255,255,255)
neo.fill_strip(220, 240, 120)
neo.update_strip()  # commit/send to LEDs

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