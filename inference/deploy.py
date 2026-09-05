
from picamera2 import Picamera2
from time import sleep
import time
from datetime import datetime 
import numpy as np
import cv2
import tflite_runtime.interpreter as tflite
import sys
import os
import pi5neo
import json
import subprocess
import psutil



img_dir = "5-25"
fps=30
lookback = 5

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
log_img_dir = os.path.join(ROOT, "logged_data", img_dir)
logs_file = os.path.join(ROOT, "logs.json")

base = os.path.join(ROOT, "models") + "/"

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
neo.fill_strip(220, 240, 120)
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

# sets up model calls for all models, returns output from model using img
def diff(diff_map):
    global old
    if diff_map == True:
        new = img
        if old is None:
                old = new
        diff = cv2.absdiff(old,new)
        print(f"Diff: {float(np.average(diff)):.6f}", end="\r\n\r\n")
        old = new
    else:
         diff = False
    return diff
def water_inference(model_path):
    preds = []
    for i in range(10): # would run for 1/3 seconds
        model = load_model(model_path["water_model"])
        interpreter = model["interpreter"]
        input_details = model["input_details"]
        output_details = model["output_details"]
        interpreter.set_tensor(input_details[0]["index"], img)
        interpreter.invoke()
        water_prediction = float(interpreter.get_tensor(output_details[0]["index"]).flat[0])
        print(f"water_prediction: {water_prediction:.6f} ------------------", end="\r\n\r\n")
        
        preds.append(water_prediction) 
        print(f"Average water prediction: {float(np.average(preds)):.6f}")
        time.sleep(1/30)
        if i > 4:
            preds = preds[1:] # removes first value to keep avg to last 5 preds
    
    water_prediction = np.average(preds)
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
    food_prediction = float(interpreter.get_tensor(output_details[0]["index"]).flat[0])
    print(f"food_prediction: {food_prediction:.6f} ------------------", end="\r\n\r\n")
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
    poop_prediction = float(interpreter.get_tensor(output_details[0]["index"]).flat[0])
    print(f"poop_prediction: {poop_prediction:.6f} ------------------", end="\r\n\r\n")
    return poop_prediction
# gets data like ram, cpu usage/temp
def hardware_data(): # will only run on pi, due to how systems pull the data
    cpu_temp = subprocess.check_output(["vcgencmd", "measure_temp"]).decode("UTF-8") # to 8 decimal points
    cpu_usage = psutil.cpu_percent(interval=1) # measures full cpu load avg'd over one sec
    ram = psutil.virtual_memory()
    used = ram.used / 1024**2 # outputs used ram in MB
    throtled = subprocess.check_output(["vcgencmd", "get_throttled"]).decode("UTF-8") #VCGENMD is the pi os system, if non zero pi is throttling
    print(f"CPU Temp: {cpu_temp.strip()} | CPU Usage: {cpu_usage}% | RAM Used: {used:.2f} MB | FPS: {fps} Throttled: {throtled.strip()}", end="\r\n\r\n")
    return cpu_temp, cpu_usage, used, throtled

old_water= not True
old_food = not True
old_poop = not True

# gets old content from logs file
with open(logs_file, "r") as f:
     content = f.read().strip()
     logs = json.loads(content) if content else []
old_logs = "start"
x = 0 # keeps track of iterations
water_list = []
food_list = []
poop_list = []
try:
    while True:
        x += 1
        log_entry = None
        new_logs = []
        if x % 5 == 0: # % is mod operator
            hardware_data()
        take_pic()

        food_prediction = None
        poop_prediction = None

        # gets water pred from function
        water_list.append(water_inference(model_path))
        food_list.append(food_inference(model_path, diff, diff_map))
        poop_list.append(poop_inference(model_path, diff, diff_map))

        if x > lookback: # lookback changes how far we go back to avg
            water_list = water_list[1:] # removes first value to keep avg to last 5 preds
            food_list = food_list[1:]
            poop_list = poop_list[1:]
        
        water_prediction = np.average(water_list)
        food_prediction = np.average(food_list)
        poop_prediction = np.average(poop_list)

        if water_prediction > 0.6:
            print(f"Water Detected: {water_prediction:.6f}", end="\r\n\r\n")
            water_presence = True
        else:
                print(f"No water detected: {water_prediction:.6f}", end="\r\n\r\n")
                water_presence = False

        
        file=logs_file
        with open(file, "r") as f:
            content = f.read().strip()
            old_logs = json.loads(content) if content else ["start"]

        if water_presence == True and old_water is not True:
            imgname= f"{log_img_dir}/{datetime.now().astimezone().strftime('%Y-%m-%d %H-%M-%S %Z')}.jpg"
            save_img = (img[0] * 255).astype(np.uint8)
            cv2.imwrite(imgname, save_img) # saves image with timestamp, can be used for future training data
            print("saved img", imgname, end="\r\n\r\n")
            log_entry=({
                "timestamp": datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z'),
                "confidence": float(water_prediction),
                "water_start": True,
                "water_end": False,
                "filepath": imgname
            })
        if water_presence is not True and old_water == True:
            imgname= f"{log_img_dir}/{datetime.now().astimezone().strftime('%Y-%m-%d %H-%M-%S %Z')}.jpg"
            save_img = (img[0] * 255).astype(np.uint8)
            cv2.imwrite(imgname, save_img  ) # saves image with timestamp, can be used for future training data
            print("saved img", imgname, end="\r\n\r\n")
            log_entry=({
                "timestamp": datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z'),
                "confidence": float(water_prediction),
                "water_start": False,
                "water_end": True,
                "filepath": imgname
            })

        if food_prediction is not None and food_prediction >.5  and old_food is not None and not old_food > .5:
            imgname= f"{log_img_dir}/{datetime.now().astimezone().strftime('%Y-%m-%d %H-%M-%S %Z')}.jpg"
            save_img = (img[0] * 255).astype(np.uint8)
            cv2.imwrite(imgname, save_img  ) # saves image with timestamp, can be used for future training data
            print("saved img", imgname, end="\r\n\r\n")
            log_entry=({
                "timestamp": datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z'),
                "confidence": float(food_prediction),
                "food_start": True,
                "food_end": False,
                "filepath": imgname
            })
        if food_prediction is not None and food_prediction < 0.5 and old_food is not None and not old_food < .5:
            imgname= f"{log_img_dir}/{datetime.now().astimezone().strftime('%Y-%m-%d %H-%M-%S %Z')}.jpg"
            save_img = (img[0] * 255).astype(np.uint8)
            cv2.imwrite(imgname, save_img  ) # saves image with timestamp, can be used for future training data
            print("saved img", imgname, end="\r\n\r\n") 
            log_entry=({
                "timestamp": datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z'),
                "confidence": float(food_prediction),
                "food_start": False,
                "food_end": True,
                "filepath": imgname
            })
        if poop_prediction is not None and poop_prediction >.5  and old_poop is not None and not old_poop > .5:
            imgname= f"{log_img_dir}/{datetime.now().astimezone().strftime('%Y-%m-%d %H-%M-%S %Z')}.jpg"
            save_img = (img[0] * 255).astype(np.uint8)
            cv2.imwrite(imgname, save_img  ) # saves image with timestamp, can be used for future training data
            print("saved img", imgname, end="\r\n\r\n")
            log_entry=({
                "timestamp": datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z'),
                "confidence": float(poop_prediction),
                "poop_start": True,
                "poop_end": False,
                "filepath": imgname
            })
        if poop_prediction is not None and poop_prediction < 0.5 and old_poop is not None and not old_poop < .5:
            imgname= f"{log_img_dir}/{datetime.now().astimezone().strftime('%Y-%m-%d %H-%M-%S %Z')}.jpg"
            save_img = (img[0] * 255).astype(np.uint8)
            cv2.imwrite(imgname, save_img) # saves image with timestamp, can be used for future training data
            print("saved img", imgname, end="\r\n\r\n")
            log_entry=({
                "timestamp": datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z'),
                "confidence": float(poop_prediction),
                "poop_start": False,
                "poop_end": True,
                "filepath": imgname
            })
        if log_entry is not None:
            logs.append(log_entry)
            new_logs.append(log_entry)
        old_water = water_presence
        old_food = food_prediction
        old_poop = poop_prediction

        if logs != old_logs or x == 2:
            if x == 2:
                imgname= f"{log_img_dir}/{datetime.now().astimezone().strftime('%Y-%m-%d %H-%M-%S %Z')}.jpg"
                save_img = (img[0] * 255).astype(np.uint8)
                cv2.imwrite(imgname, save_img) # saves image with timestamp, can be used for future training data
                print("saved img", imgname, end="\r\n\r\n")
                log_entry=({
                    "timestamp": datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z'),
                    "water_confidence": float(water_prediction),
                    "food_confidence": float(food_prediction) if food_prediction is not None else 0.0,
                    "poop_confidence": float(poop_prediction) if poop_prediction is not None else 0.0,
                    "filepath": imgname,
                    "first_log": True
                })
                logs.append(log_entry)
                new_logs.append(log_entry)
            print(new_logs, end="\r\n\r\n")
            with open(file, "w") as f:
                    json.dump(logs, f, indent=4)
            old_logs = logs.copy()
        
        time.sleep(.5)
except KeyboardInterrupt:
    picam2.stop()
    neo.fill_strip(0, 0, 0)
    neo.update_strip()