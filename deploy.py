from picamera2 import Picamera2
from time import sleep
import time
import numpy as np
import cv2
import tflite_runtime.interpreter as tflite
import sys
import os
from pi5neo import Pi5Neo
import json
import subprocess
import psutil

file="logs.json"
with open(file, "w") as f:
            json.dump(["start"], f)

interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

#sets up cam
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (224, 224)}, buffer_count=4)
picam2.configure(config)
picam2.start()

#sets up led ring
SPI_DEVICE = '/dev/spidev0.0' # Rpi protocol to get the timing right for the GPIOs
SPI_SPEED_KHZ = 800 #speed of SPI protocol

neo = Pi5Neo(SPI_DEVICE, 30, SPI_SPEED_KHZ)

# Fill the strip with white (R,G,B = 255,255,255)
neo.fill_strip(255, 255, 255)
neo.update_strip()  # commit/send to LEDs


def take_pic():
    frame = picam2.capture_array()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #the pi cam takes in BGR
    img = cv2.resize(img, (224, 224))
    # Normalize
    img = img.astype("float32") / 255.0
    # Add batch dimension → (1, 224, 224, 3)
    img = np.expand_dims(img, axis=0)
    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()
    # gets prediciton
    prediction = interpreter.get_tensor(output_details[0]["index"])
    # Extract the single float value from the nested array
    prediction = prediction[0][0]

    # Print the raw value formatted to 8 decimal places for readability
    print(f"{prediction:.8f}")
    return prediction
water_on= False
def hardware_data():
    cpu_temp = subprocess.check_output(["vcgencmd", "measure_temp"]).decode("UTF-8") # to 8 decimal points
    cpu_usage = psutil.cpu_percent(interval=1) # measures full cpu load avg'd over one sec
    ram = psutil.virtual_memory()
    used = ram.used / 1024**2 # outputs used ram in MB
    throtled = subprocess.check_output(["vcgencmd", "get_throttled"]).decode("UTF-8") #VCGENMD is the pi os system, if non zero pi is throttling
    print (f"CPU Temp: {cpu_temp.strip()} | CPU Usage: {cpu_usage}% | RAM Used: {used:.2f} MB | Throttled: {throtled.strip()}")
    return cpu_temp, cpu_usage, used, throtled
x= 0
while x < 100: # runs model 10 times
    sleep(0.2)
    start = time.perf_counter()
    prediction = take_pic()
    end = time.perf_counter()
    print ((end-start)*1000) #acounts for data aquisition and infernce, which seems more usefull
    x += 1
    data = None  # Initialize data to None at the start of the loop
    # bucket logic
    if prediction >= 0.6:
        water_on = True
        data = {
            "water_starts": True,
            "Time": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        print("Water Started")
    elif prediction <= 0.4:
        water_on = False
        data = { # data for json
            "water_stops": True,
            "Time": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        print("Water Stopped")
    else:
        print("Either no data to log or model not confident enough.")

# logs data
    if data:
        with open(file, "r") as f:
            logs = json.load(f)
            logs = [] # wipes file at run time for testing
        logs.append(data)
        with open(file, "w") as f:
            json.dump(logs, f, indent=4) # adds current log
    if x % 10 == 0: # runs the hardware logs every 2 seconds
        hardware_data()
neo.fill_strip(0, 0, 0)
neo.update_strip()  # send to LEDs