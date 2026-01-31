import time
import numpy as np
import sys
import cv2
import os
import json
import tensorflow as tf
from PIL import Image
from pi5neo import Pi5Neo
from picamera2 import Picamera2

model = "/Users/jakehopkins/Downloads/if_water/clean_dirty_data_aug20260130_105338.keras"
file = "logs.json"
test= [3,2,1]
order = ["clean", "food", "no_water", "toilet"]
img_dir = "/Users/jakehopkins/Downloads/if_water/Clean_Dirty/test"  # Directory with images

#hardware setup
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (224, 224)}, buffer_count=4)
picam2.configure(config)
picam2.start()

SPI_DEVICE = '/dev/spidev0.0' # Rpi protocol to get the timing right for the GPIOs
SPI_SPEED_KHZ = 800 #speed of SPI protocol

neo = Pi5Neo(SPI_DEVICE, 30, SPI_SPEED_KHZ)

neo.fill_strip(255, 255, 255)
neo.update_strip()  # commit/send to LEDs

def prediciton(img_path):
    global img, img_array
    frame = picam2.capture_array()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #the pi cam takes in BGR
    img = cv2.resize(img, (224, 224))

    interpreter = tf.lite.Interpreter(model_path=model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Load and preprocess image
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]["index"])
    prediction = [round(x, 2) for x in prediction[0]]

    prediction = np.argmax(prediction)
    pred = order[prediction]
    print(f"{img_path}: {pred} {prediction}")
    return pred

current_pred = None
start = time.time()
def log(pred):

    water_status= str(pred) # returns string of prediction, like clean, food, etc.
    print(water_status)
    now = time.time() 
    global current_pred, start, time_on, start_time
    if current_pred == None:
        current_pred = pred
        start_time = time.time()
        return 
    if pred == current_pred:
        run_length += 1
    if pred != current_pred:
        time_on = now - start_time
        current_pred = pred
        start_time = now
    time_on = np.round(time_on, 3) # rounds to 2 numbers after the decimal
    # gets old data and adds it to list
    try:
        with open(file, "r") as f:
            old = json.load(f)
            ID = [e["ID"] for e in old if isinstance(e, dict) and "ID" in e]
            ID = max(ID) + 1 if ID else 1
    except(json.JSONDecodeError):
        old = []
        ID = 1

# edge case for empty file
    if not isinstance(old, list): 
        old = test

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

result = prediciton(img)
    
if result != previous_prediction:
    log(result)
    previous_prediction = result