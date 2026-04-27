from picamera2 import Picamera2
from time import sleep
import time
import numpy as np
import cv2
import tflite_runtime.interpreter as tflite
import sys
import os
import pi5neo
import json
import subprocess
import psutil

fps=30

base = "/home/jake/Downloads/if-water-cnn/models/"

model_path = {
     "food_model":  base + "food_full_diff.tflite",
     "water_model": base + "if_water.tflite",
     "poop_model": base + "poop_model.tflite"
}

diff_map = {
     "food_model": True,
     "water_model": False,
     "poop_model": True

}


def load_model(model_path):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()        
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return {
        "interpreter": interpreter,
        "input_details": input_details,
        "output_details": output_details
    }


picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (224, 224)}, buffer_count=4)
picam2.configure(config)
picam2.start()

#sets up led ring
SPI_DEVICE = '/dev/spidev0.0' # Rpi protocol to get the timing right for the GPIOs
SPI_SPEED_KHZ = 800 #speed of SPI protocol

neo = pi5neo.Pi5Neo(SPI_DEVICE, 30, SPI_SPEED_KHZ)

# Fill the strip with white (R,G,B = 255,255,255)
neo.fill_strip(255, 255, 255)
neo.update_strip()  # commit/send to LEDs


def take_pic():
    global img
    frame = picam2.capture_array()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #the pi cam takes in BGR
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32") / 255.0
    # Add batch dimension → (1, 224, 224, 3)
    img = np.expand_dims(img, axis=0)
    return img
old = None
def diff(diff_map):
    global old
    if diff_map == True:
        new = img
        if old is None:
                old = new
        diff = cv2.absdiff(old,new)
        print(np.average(diff))
        old = new
    else:
         diff = False
    return diff
def water_inference(model_path, diff, diff_map): 
    model = load_model(model_path["water_model"])
    diff = diff(diff_map["water_model"])
    interpreter = model["interpreter"]
    input_details = model["input_details"]
    output_details = model["output_details"]
    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()
    water_prediction = interpreter.get_tensor(output_details[0]["index"])
    print(water_prediction)
    return water_prediction
def food_inference(model_path, diff, diff_map):
    model = load_model(model_path["food_model"])
    diff = diff(diff_map["food_model"])
    interpreter = model["interpreter"]
    input_details = model["input_details"]
    output_details = model["output_details"]
    arr = np.concatenate([img, diff], axis=-1)
    interpreter.set_tensor(input_details[0]["index"], arr)
    interpreter.invoke()
    food_prediction = interpreter.get_tensor(output_details[0]["index"])
    print("food_prediction:", food_prediction)
    return food_prediction
def poop_inference(model_path, diff, diff_map):
    model = load_model(model_path["poop_model"])
    diff = diff(diff_map["poop_model"])
    interpreter = model["interpreter"]
    input_details = model["input_details"]
    output_details = model["output_details"]
    arr = np.concatenate([img, diff], axis=-1)
    interpreter.set_tensor(input_details[0]["index"], arr)
    interpreter.invoke()
    poop_prediction = interpreter.get_tensor(output_details[0]["index"])
    print("poop_prediction:", poop_prediction)
    return poop_prediction

def hardware_data(): # will only run on pi, due to how systems pull the data
    cpu_temp = subprocess.check_output(["vcgencmd", "measure_temp"]).decode("UTF-8") # to 8 decimal points
    cpu_usage = psutil.cpu_percent(interval=1) # measures full cpu load avg'd over one sec
    ram = psutil.virtual_memory()
    used = ram.used / 1024**2 # outputs used ram in MB
    throtled = subprocess.check_output(["vcgencmd", "get_throttled"]).decode("UTF-8") #VCGENMD is the pi os system, if non zero pi is throttling
    print (f"CPU Temp: {cpu_temp.strip()} | CPU Usage: {cpu_usage}% | RAM Used: {used:.2f} MB | FPS: {fps} Throttled: {throtled.strip()}")
    return cpu_temp, cpu_usage, used, throtled

old_water= None
old_food = None
old_poop = None

logs = []
while True:
    take_pic()

    water_prediction = water_inference(model_path, diff, diff_map)
    if water_prediction > 0.5:
        print("Water Detected", water_prediction)
        water_presence = True
    else:
            print("no water detected", water_prediction)
            water_presence = False

    if water_presence == True:
        food_prediction = food_inference(model_path, diff, diff_map)
        poop_prediction = poop_inference(model_path, diff, diff_map)

    sleep(.5)
    
    file="logs.json"
    with open(file, "r") as f:
        content = f.read().strip()
        old_logs = json.loads(content) if content else ["start"]
    print(logs)
    if water_presence == True and old_water == False:
        imgname= f"logged_data/water_start{int(time.time())}.jpg"
        cv2.imsave(img, imgname) # saves image with timestamp, can be used for future training data
        logs.append({
             "timestamp": time.time(),
             "confidence": float(water_prediction),
             "water_start": True,
             "water_end": False,
             "filepath": imgname
        })
    if water_presence == False and old_water == True:
        imgname= f"logged_data/water_end{int(time.time())}.jpg"
        cv2.imsave(img, imgname) # saves image with timestamp, can be used for future training data
        logs.append({
             "timestamp": time.time(),
             "confidence": float(water_prediction),
             "water_start": False,
             "water_end": True,
             "filepath": imgname
        })

    if food_prediction >.5  and old_food == False:
        imgname= f"logged_data/food_start{int(time.time())}.jpg"
        cv2.imsave(img, imgname) # saves image with timestamp, can be used for future training data
        logs.append({
             "timestamp": time.time(),
             "confidence": float(food_prediction),
             "food_start": True,
             "food_end": False,
             "filepath": imgname
        })
    if food_prediction < 0.5 and old_food == True:
        imgname= f"logged_data/food_end{int(time.time())}.jpg"
        cv2.imsave(img, imgname) # saves image with timestamp, can be used for future training data
        logs.append({
             "timestamp": time.time(),
             "confidence": float(food_prediction),
             "food_start": False,
             "food_end": True,
             "filepath": imgname
        })
    if poop_prediction >.5  and old_poop == False:
        imgname= f"logged_data/poop_start{int(time.time())}.jpg"
        cv2.imsave(img, imgname) # saves image with timestamp, can be used for future training data
        logs.append({
             "timestamp": time.time(),
             "confidence": float(poop_prediction),
             "poop_start": True,
             "poop_end": False,
             "filepath": imgname
        })
    if poop_prediction < 0.5 and old_poop == True:
        imgname= f"logged_data/poop_end{int(time.time())}.jpg"
        cv2.imsave(img, imgname) # saves image with timestamp, can be used for future training data
        logs.append({
             "timestamp": time.time(),
             "confidence": float(poop_prediction),
             "poop_start": False,
             "poop_end": True,
             "filepath": imgname
        })
    old_water = water_presence
    old_food = food_prediction
    old_poop = poop_prediction

    if logs != old_logs:
        with open(file, "w") as f:
                json.dump(logs, f)
        old_logs = logs
    
