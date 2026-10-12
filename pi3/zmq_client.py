#subnet ip for pi3: 10.10.10.2
import time
import zmq


context = zmq.Context()

#  Socket to talk to server
print("Connecting to hello world server...")
socket = context.socket(zmq.REQ)
socket.connect("tcp://10.10.10.2:5555")

#  Do 10 requests, waiting each time for a response
x=1
for request in range(10):
    print(f"Sending request {request} ...")
    socket.send_string(f"sent from pi3 message numer{x}")
    x+=1

    #  Get the reply.
    message = socket.recv()
    print(f"Received reply {request} [ {message} ]")