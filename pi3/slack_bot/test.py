from slack_sdk import WebClient
from dotenv import load_dotenv
import os
import zmq
import cv2 as cv
import numpy as np
import time

load_dotenv()
client = WebClient(token=os.environ["SLACK_BOT_TOKEN"])
dm = client.conversations_open(users="U08KTKDSADT")#your member ID

clean_img=cv.imread("img.jpg")

context = zmq.Context()
socket=context.socket(zmq.REQ)
socket.connect("tcp://pi3.local:5555")
old = None#diff from the last loop
while True:
    socket.send_string("image")
    jpg, stamp = socket.recv_multipart()

    img = cv.imdecode(np.frombuffer(jpg, np.uint8), cv.IMREAD_COLOR)#jpg bytes -> image array
    diff = np.average(cv.absdiff(cv.cvtColor(img, cv.COLOR_BGR2GRAY), cv.cvtColor(clean_img, cv.COLOR_BGR2GRAY)))

    if diff>3:
        print("stuff in image")
        if old is not None and abs(diff-old)<5: # image hasn't changed in a bit
            client.chat_postMessage(channel=os.environ["SLACK_USER_ID"], text="Warning, there might be a clog")
            client.files_upload_v2(
                    channel=dm["channel"]["id"],
                    content=jpg,
                    filename="img.jpg",
                )
            print("send error warning")
    old = diff
    print (diff)

    # cv.imshow("image", img)
    # cv.imwrite("img.jpg", img)
    # cv.waitKey(0)#waits for a key press
    # cv.destroyAllWindows()
    
    time.sleep(60)
