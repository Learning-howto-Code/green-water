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

def hardware_data(): # will only run on pi, due to how systems pull the data
    cpu_temp = subprocess.check_output(["vcgencmd", "measure_temp"]).decode("UTF-8") # to 8 decimal points
    cpu_usage = psutil.cpu_percent(interval=1) # measures full cpu load avg'd over one sec
    ram = psutil.virtual_memory()
    used = ram.used / 1024**2 # outputs used ram in MB
    throtled = subprocess.check_output(["vcgencmd", "get_throttled"]).decode("UTF-8") #VCGENMD is the pi os system, if non zero pi is throttling
    print (f"CPU Temp: {cpu_temp.strip()} | CPU Usage: {cpu_usage}% | RAM Used: {used:.2f} MB | Throttled: {throtled.strip()}")
    return cpu_temp, cpu_usage, used, throtled
x = 0 
for x in range(100):
    hardware_data()
    sleep(1)
    x += 1