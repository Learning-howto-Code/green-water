import zmq
import time
import gpiozero
import board
import neopixel_spi as neopixel # type: ignore
from picamera2 import Picamera2
import cv2 as cv
from datetime import datetime
#uses GPIO23(16)
# solenoid= gpiozero.LED(16)
# solenoid.on()
# solenoid.off()


#sets up neopixel ring
pixels = neopixel.NeoPixel_SPI(
    board.SPI(), 24, brightness=1.0, auto_write=False, pixel_order=neopixel.GRB
)
pixels.fill((220, 240, 120))
pixels.show()

time.sleep(1)
#sets up camera
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (640, 360)}, buffer_count=4)
picam2.configure(config)
picam2.start()
time.sleep(1)
ctx=zmq.Context()
sock=ctx.socket(zmq.REP)#answers when pi asks
sock.bind("tcp://*:5555")
sock.bind("ipc://*:5555 ")

time.sleep(1)#must wait for everything to start up, otherwise colors are wonky

def take_img():# sends image and time
    img = picam2.capture_array()
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.rotate(img, cv.ROTATE_180)
    success, jpg = cv.imencode('.jpg', img)
    stamp = str(datetime.now().strftime("%H-%M-%S-%f")[:-3])
    return img, stamp
# open ranges from 135-145
#closed ranges from 235-238
def valve_check():
    img, stamp =take_img()
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    mean = cv.mean(img[0:55, 300:375])[0]
    if mean >235:
        valve_true = "closed"
    elif mean<145:
        valve_true = "open"
    else:
        valve_true="Error"
    return valve_true
timer = time.monotonic()
# def move_valve(state):# sends string
#     valve_true = valve_check()
#     if (time.monotonic()) - timer > (4*60*60):
#         # valve needs to close
#         solenoid.off()
#         state = "closed" 
#     if state == "open" and valve_true == "closed":
#         solenoid.on()
#         state="open"
#     if state == "closed" and valve_true =="open":
#         solenoid.off()
#         state="closed"
#     time.sleep(1)
#     valve_true = valve_check()
#     if valve_true != state:
#         sock.send_string(f"Error {valve_true}")
#         return f"Error {valve_true}"
#     else:
#         sock.send_string(valve_true)
#         return valve_true

try:
    while True:
        message = sock.recv().decode()#waits for request
        if "image" in message:
            img, stamp = take_img()
            success, jpg = cv.imencode('.jpg', img)
            sock.send_multipart([jpg.tobytes(), stamp.encode()])
        # if "open"in message or "closed" in message:
        #     move_valve(message)
        # time.sleep(.5)
        
finally:
    sock.close()
    picam2.close()
    pixels.deinit()
    # solenoid.off()




