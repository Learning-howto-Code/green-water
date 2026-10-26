#RunAWaterModelConstantly
#logs every second there's water
#
from calendar import c
import os

import tflite_runtime.interpreter as tflite # type: ignore
import time
import datetime
import cv2 
from picamera2 import Picamera2 # type: ignore
import board
import neopixel_spi as neopixel
import numpy as np
import psutil
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
model = os.path.join(ROOT, "models/if_water.tflite")
dir = os.path.join(ROOT, "data/6-15")
os.makedirs(dir, exist_ok=True)
lookback = 5
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (224, 224)}, buffer_count=4)
picam2.configure(config)
picam2.start()

#sets up led ring
pixels = neopixel.NeoPixel_SPI(
    board.SPI(), 30, brightness=1.0, auto_write=False, pixel_order=neopixel.GRB
)
pixels.fill((220, 240, 120))
pixels.show()  # commit/send to LEDs

def take_pic():
    global img
    frame = picam2.capture_array()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #the pi cam takes in BGR
    raw = img.copy()
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32") / 255.0
    # Add batch dimension → (1, 224, 224, 3)
    img = np.expand_dims(img, axis=0)
    return img, raw

interpreter = tflite.Interpreter(model_path=model)
interpreter.allocate_tensors()        
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
def run_model(img):
    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()
    water_prediction = float(interpreter.get_tensor(output_details[0]["index"]).flat[0])

    return water_prediction

def hardware_data(): # will only run on pi, due to how systems pull the data
    cpu_temp = subprocess.check_output(["vcgencmd", "measure_temp"]).decode("UTF-8") # to 8 decimal points
    cpu_usage = psutil.cpu_percent(interval=1) # measures full cpu load avg'd over one sec
    ram = psutil.virtual_memory()
    used = ram.used / 1024**2 # outputs used ram in MB
    throtled = subprocess.check_output(["vcgencmd", "get_throttled"]).decode("UTF-8") #VCGENMD is the pi os system, if non zero pi is throttling
    print(f"CPU Temp: {cpu_temp.strip()} | CPU Usage: {cpu_usage}% | RAM Used: {used:.2f} MB | FPS: {1} Throttled: {throtled.strip()}", end="\r\n\r\n")
    return cpu_temp, cpu_usage, used, throtled


pred_list = []
seconds_on = 0
total_seconds = 0
with open(os.path.join(ROOT, "seconds_on.txt"), "w") as f:
    f.write("First log on run")
    
img, raw = take_pic()
cv2.imwrite(f"{dir}/first_run.jpg", raw)
print(f"Saved image to {dir}")
while True:
    img, raw = take_pic()
    water_prediction = run_model(img)
    
    pred_list.append(water_prediction)

    if len(pred_list) > lookback:
        pred_list = pred_list[1:]
    water_prediction = np.average(pred_list)
    
    if water_prediction > 0.6:
        print(f"water detected: {str(water_prediction)}")
        print(f"time on is {str(seconds_on)} + 1 seconds")
        seconds_on += 1
    else:
        print(f"no water detected: {str(water_prediction)}")

    if seconds_on % 50 == 0:
        with open(os.path.join(ROOT, "seconds_on.txt"), "w") as f:
            f.write(f"Seconds on: {str(seconds_on)} out of {str(total_seconds)} seconds\n")
            print("logging seconds on to file")
        cv2.imwrite(f"{dir}/{water_prediction}.jpg", raw)
        print(f"Saved image to {dir}")
    if total_seconds % 10 == 0:
        hardware_data()
        print(f"time is {datetime.datetime.now()}")
    total_seconds += 1
    time.sleep(1)

