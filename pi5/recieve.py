#subnet ip for pi5: 10.10.10.1, pi3:10.10.10.2
import re
import time
import gpiozero
import zmq
import cv2 as cv
import numpy as np
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://10.10.10.2:5555")

solenoid= gpiozero.LED(16)
solenoid.on()
solenoid.off()
    
x=1
def get_img():
    socket.send_string(f"need image {x}")
    jpg, stamp = socket.recv_multipart()
    print(stamp)
    jpg = np.frombuffer(jpg, np.uint8)
    jpg = cv.imdecode(jpg, cv.IMREAD_COLOR)
    return jpg
state = "closed"
def move_valve(state):
    socket.send_string(state)
    state = socket.recv_string()
while True:
    jpg=get_img()
    cv.imshow("image", jpg)
    cv.waitKey(1)
    cv.destroyAllWindows
    x+=1
    time.sleep(1)