import time
import board
import zmq

import zmq

context = zmq.Context()

#  Socket to talk to server
print("Connecting to hello world server...")
socket = context.socket(zmq.REQ)
socket.connect("tcp://100.127.41.73:5555")

#  Do 10 requests, waiting each time for a response
x=1
for request in range(10):
    print(f"Sending request {request} ...")
    socket.send_string("sent from pi3 message numer"+x)
    x+=1

    #  Get the reply.
    message = socket.recv()
    print(f"Received reply {request} [ {message} ]")