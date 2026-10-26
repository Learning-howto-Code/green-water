from datetime import datetime
import neopixel_spi as neopixel # type: ignore
import board
from picamera2 import Picamera2
import cv2 as cv
import time

pixels = neopixel.NeoPixel_SPI(
    board.SPI(), 24, brightness=0.2, auto_write=False, pixel_order=neopixel.RGB
)
pixels.fill((220, 240, 120))
pixels.brightness=.2
pixels.show()  # commit/send to LEDs

picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (1280, 720)}, buffer_count=4)
picam2.configure(config)
picam2.start()


time.sleep(1)

for i in range(10):
    img = picam2.capture_array()
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.rotate(img, cv.ROTATE_180)
    stamp = str(datetime.now().strftime("%H-%M-%S-%f")[:-3])
    cv.imwrite(f"data/{stamp}.jpg", img)
    time.sleep(.2)