import zmq
import time
import gpiozero
import board
import neopixel
from picamera2 import Picamera2
import cv2 as cv
from datetime import datetime
#uses GPIO23(16)
solenoid= gpiozero.LED(16)
solenoid.on()
solenoid.off()


#sets up neopixel ring
neopixel_pin=board.D10
pixels = neopixel.NeoPixel(
    neopixel_pin, 24, brightness=0.2, auto_write=False, pixel_order=neopixel.RGB
)
#sets up camera
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (640, 360)}, buffer_count=4)
picam2.configure(config)
picam2.start()

ctx=zmq.Context()
sock=ctx.socket(zmq.REP)#answers when pi asks
sock.bind("tcp://100.92.80.61:5555")

time.sleep(1)#must wait for everything to start up, otherwise colors are wonky

def take_img():
    img = picam2.capture_array()
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    success, jpg = cv.imencode('.jpg', img)
    stamp = str(datetime.now().strftime("%H-%M-%S-%f")[:-3])
    return jpg.tobytes(), stamp
while True:
    sock.recv()#waits for request
    jpg, stamp = take_img()
    sock.send_multipart([jpg, stamp.encode()])




