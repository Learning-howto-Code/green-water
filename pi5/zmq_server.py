#subnet ip for pi5: 10.10.10.1
import time
import zmq
import cv2 as cv
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.bind("tcp://*:5555")
x=1
while True:
    socket.send_string(f"need image {x}")
    jpeg, stamp = socket.recv()
    print(stamp)
    jpg = cv.imdecode(jpeg, cv.IMREAD_COLOR)
    cv.imshow("image", jpg)
    cv.waitKey(1)
    cv.destroyAllWindows
    x+=1