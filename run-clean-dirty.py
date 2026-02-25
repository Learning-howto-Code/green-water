import datetime
import time
import numpy as np
import cv2 as cv
import json
import tflite_runtime.interpreter as tflite
from PIL import Image
from pi5neo import Pi5Neo
from picamera2 import Picamera2

seconds = int(20) #how long to run the script for.

if_water_model = "if_water.tflite"
food_clean_model = "food_clean.tflite"
file = "logs.json"

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
old_file = None
def if_water():
    global img, img_array,frame, pred
    frame = picam2.capture_array()
    img = cv.cvtColor(frame, cv.COLOR_BGR2RGB) #the pi cam takes in BGR
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + f"-{datetime.datetime.now().microsecond // 1000:03d}"

    img_name = f"logged_data/ID#{ID}, {timestamp}.jpg"
    cv.imwrite(img_name, img)
    img = cv.resize(img, (224, 224))

    interpreter = tflite.Interpreter(model_path=if_water_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)
    
    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]["index"])
    pred_int = prediction
    pred_int = np.round(pred_int, 4) #rounds pred_int to 4 points
    prediction = [round(x, 2) for x in prediction[0]]
    if pred_int > 0.5:
        pred = "water"
    if pred_int <= 0.5:
        pred = "no water"
    print(pred_int, pred)

current_pred = None
start = time.time()
def food_clean():
    global img, img_array,frame, pred
    frame = picam2.capture_array()
    img = cv.cvtColor(frame, cv.COLOR_BGR2RGB) #the pi cam takes in BGR


    img = cv.resize(img, (224, 224))

    if old_file is None:
        old_file = img
    new_file = img

    #diff calculation
    old = cv.imread(old_file )     
    new_img = cv.imread(new_file)
    diff = cv.absdiff(old, new_img)
    diff = np.average(diff)
    old_file = new_file
    

    interpreter = tflite.Interpreter(model_path=food_clean_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Load and preprocess image
    diff = np.full((224,224,), diff, dtype=np.float32)
  # Add diff as an additional channel
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.dstack((img, diff))
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]["index"])
    pred_int = prediction
    pred_int = np.round(pred_int, 4) #rounds pred_int to 4 points
    prediction = [round(x, 2) for x in prediction[0]]
    if pred_int > 0.5:
        pred = "clean"
    if pred_int <= 0.5:
        pred = "dirty"
    print(pred_int, pred)
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

    try: 
        with open(file, "r") as f: #gets highest ID number
            old = json.load(f)
            ID = [e["ID"] for e in old if isinstance(e, dict) and "ID" in e]
            ID = max(ID) + 1 if ID else 1 # first run condition
    except(json.JSONDecodeError):
        old = []
        ID = 1

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
get_ID()
for x in range(seconds*30): #for 30 fps
    if_water()
    food_clean()
    if pred != previous_prediction: # logic for when to run the logging funtion
        log(pred)
        print(pred)
        previous_prediction = pred
time.sleep(0.5)
neo.fill_strip(0, 0, 0)
neo.update_strip()  # commit/send to LEDs
