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
def get_img(): # sends string, expects image and time
    socket.send_string(f"need image {x}")
    jpg, stamp = socket.recv_multipart()
    print(stamp)
    jpg = np.frombuffer(jpg, np.uint8)
    jpg = cv.imdecode(jpg, cv.IMREAD_COLOR)
    return jpg
state = "closed"
def move_valve(state):# sends string, expects string
    socket.send_string(state)
    print("send state")
    state = socket.recv_string() 
    print("recieved state:", state)
while True:# as REQ must send first
    move_valve(state) # first sends state on run, expects string back
    jpg=get_img()
    cv.imwrite(f"data/{x}.jpg", jpg)
    print("saved image")
    # cv.imshow("image", jpg)
    # cv.waitKey(1)
    # cv.destroyAllWindows
    x+=1
    time.sleep(1)
    if state == "closed":
        state = "open"
    else:
        state = "closed"