from picamera2 import Picamera2 # type: ignore
from time import sleep
import time
from datetime import date, datetime
import numpy as np
import cv2 as cv
import tflite_runtime.interpreter as tflite # type: ignore
import sys
import os
import pi5neo  # type: ignore
import json
import subprocess
import psutil
#vars
dir = f"data/{date.today()}"
model = "models/if_water.tflite"

#turns on light
SPI_DEVICE = '/dev/spidev0.0' # Rpi protocol to get the timing right for the GPIOs
SPI_SPEED_KHZ = 800 #speed of SPI protocol

neo = pi5neo.Pi5Neo(SPI_DEVICE, 24, SPI_SPEED_KHZ) #Pins 5v=2, GND=6, DIN=19

neo.fill_strip(255, 255, 255)
neo.update_strip()  # commit/send to LEDs
print("light on")
#instantiates camera
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (224, 224)}, buffer_count=4)
picam2.configure(config)
picam2.start()

#load model
interpreter = tflite.Interpreter(model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def take_pic():
    global pre
    frame = picam2.capture_array()
    img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    pre = img
    img = cv.resize(img, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img
def water_inference(img):
    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]["index"])
    return prediction
def save_img(img, prediction):
    os.makedirs(dir, exist_ok=True)
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]   # ms
    filename = f"{dir}/{time}_pred_{prediction[0][0]:.4f}.jpg"
    cv.imwrite(filename, img)

try:
    while True:
        img = take_pic()
        prediction = water_inference(img)
        if prediction > 0.5: # means guesses water
            color= (0, 255, 0) # green
        else:
            color = (0, 0, 255) # red
        pre = cv.cvtColor(pre, cv.COLOR_BGR2RGB)
        h = pre.shape[0]
        cv.rectangle(pre, (0, 0), (h, h), color, 2)
        save_img(pre, prediction)
        print("taking pic")
        time.sleep(1)
finally:
    picam2.close()
    neo.clear_strip()
    neo.update_strip()
    print("\n cleanly shut down")
