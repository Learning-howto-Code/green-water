import datetime
import time
import numpy as np
import sys
import cv2
import os
import json
import tflite_runtime.interpreter as tflite
from PIL import Image
from pi5neo import Pi5Neo
from picamera2 import Picamera2

seconds = int(20) #how long to run the script for.


model = "clean_dirty.tflite"
file = "logs.json"
order = ["clean", "food", "no_water", "toilet"]

#hardware setup
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (640, 360)}, buffer_count=4)
picam2.configure(config)
picam2.start()

SPI_DEVICE = '/dev/spidev0.0' # Rpi protocol to get the timing right for the GPIOs
SPI_SPEED_KHZ = 800 #speed of SPI protocol

neo = Pi5Neo(SPI_DEVICE, 30, SPI_SPEED_KHZ)

neo.fill_strip(220, 240, 120)
neo.update_strip()  # commit/send to LEDs
time.sleep(0.5)

def get_ID():
    global ID
    try: 
        with open(file, "r") as f: #gets highest ID number
            old = json.load(f)
            ID = [e["ID"] for e in old if isinstance(e, dict) and "ID" in e]
            ID = max(ID) + 1 if ID else 1 # first run condition
    except(json.JSONDecodeError):
        old = []
        ID = 1
def prediciton():
    global img, img_array,frame, pred
    frame = picam2.capture_array()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #the pi cam takes in BGR
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + f"-{datetime.datetime.now().microsecond // 1000:03d}"

    img_name = f"logged_data/ID#{ID}, {timestamp}.jpg"
    cv2.imwrite(img_name, img)
    img = cv2.resize(img, (224, 224))

    interpreter = tflite.Interpreter(model_path=model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Load and preprocess image
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]["index"])
    pred_int = prediction
    print(pred_int)
    prediction = [round(x, 2) for x in prediction[0]]

    prediction = np.argmax(prediction)
    pred = order[prediction]
    return pred

current_pred = None
start = time.time()
def log(pred):

    global current_pred, start, time_on, start_time, ID

    water_status= str(pred) # returns string of prediction, like clean, food, etc.
    print(water_status)
    now = time.time() 
    
    if current_pred == None: # run time condition
        current_pred = pred
        start_time = time.time()
        return 
    
    if pred != current_pred: #runs when prediction changes, gets time from
        time_on = now - start_time
        current_pred = pred
        start_time = time.time() #rests start time
    time_on = np.round(time_on, 3) # rounds to 2 numbers after the decimal

    # try: 
    #     with open(file, "r") as f: #gets highest ID number
    #         old = json.load(f)
    #         ID = [e["ID"] for e in old if isinstance(e, dict) and "ID" in e]
    #         ID = max(ID) + 1 if ID else 1 # first run condition
    # except(json.JSONDecodeError):
    #     old = []
    #     ID = 1

# edge case for empty file
    if not isinstance(old, list): 
        old = []

    old.append({ # adds new data onto end of old data list
        "water_status": water_status,
        "time_on": str(time_on),
        "timestamp": str(time.strftime("%Y%m%d-%H%M%S")),
        "ID": ID,
    }) 
    # adds appanded list to file
    with open(file, "w") as f:
        json.dump(old, f, indent=3)

# Get all images from directory and process them

previous_prediction = None
for x in range(seconds*30): #for 30 fps
    get_ID()
    prediciton()
    
if pred != previous_prediction: # logic for when to run the logging funtion
    log(pred)
    previous_prediction = pred
time.sleep(0.5)
neo.fill_strip(0, 0, 0)
neo.update_strip()  # commit/send to LEDs
