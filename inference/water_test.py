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

iw_model = 'models/water_testing.tflite'
f_model = 'models/food_full_diff.tflite'
p_model='models/poop_model.tflite'

#turns on light
SPI_DEVICE = '/dev/spidev0.0' # Rpi protocol to get the timing right for the GPIOs
SPI_SPEED_KHZ = 800 #speed of SPI protocol

neo = pi5neo.Pi5Neo(SPI_DEVICE, 24, SPI_SPEED_KHZ) #Pins 5v=2, GND=6, DIN=19

neo.fill_strip(220, 240, 120)
neo.update_strip()  # commit/send to LEDs
time.sleep(1)
print("light on")
#instantiates camera
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (448, 448), "format":"RGB888"}, buffer_count=4)
picam2.configure(config)
picam2.start()

time.sleep(1)
iw_interpreter = tflite.Interpreter(iw_model)
iw_interpreter.allocate_tensors()
iw_input_details = iw_interpreter.get_input_details()
iw_output_details = iw_interpreter.get_output_details()

f_interpreter = tflite.Interpreter(f_model)
f_interpreter.allocate_tensors()
f_input_details = f_interpreter.get_input_details()
f_output_details = f_interpreter.get_output_details()

p_interpreter = tflite.Interpreter(p_model)
p_interpreter.allocate_tensors()
p_input_details = p_interpreter.get_input_details()
p_output_details = p_interpreter.get_output_details()


def take_pic():
    global pre
    frame = picam2.capture_array()
    img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    img = frame
    pre = img.copy()
    img = cv.resize(img, (224, 224))
    return img, pre
def water_inference(img):
    
    iw_interpreter.set_tensor(iw_input_details[0]["index"], img)
    iw_interpreter.invoke()
    iw_prediction = float(iw_interpreter.get_tensor(iw_output_details[0]["index"]).flat[0])

    return iw_prediction
def food_inference(img):
    f_interpreter.set_tensor(f_input_details[0]["index"], img)
    f_interpreter.invoke()
    f_prediction = float(f_interpreter.get_tensor(f_output_details[0]["index"]).flat[0])

    return f_prediction
def poop_inference(img):
    p_interpreter.set_tensor(p_input_details[0]["index"], img)
    p_interpreter.invoke()
    p_prediction = float(p_interpreter.get_tensor(p_output_details[0]["index"]).flat[0])

    return p_prediction
def save_img(pre, prediction, diff):
    dir = f"data/{date.today()}"
    os.makedirs(dir, exist_ok=True)
    stamp = datetime.now().strftime("%H-%M-%S-%f")[:-3]
    filename = f"{dir}/{stamp}_pred_{float(prediction):.4f}.jpg"
    diff_name = f"{dir}/{stamp}_diff_{float(np.average(diff)):.6f}.jpg"
    print(f"'Saving image' {filename}")
    cv.imwrite(filename, pre)
    cv.imwrite(diff_name, diff)
old = None
def find_diff(img):
    global old
    new = img
    if old is None:
            old = new
    diff = cv.absdiff(old,new)
    print(f"Diff: {float(np.average(diff)):.6f}", end="\r\n\r\n")
    old = new
    return diff
def normalize(img, diff):
     img = img.astype("float32") / 255.0
     diff = diff.astype("float32") / 255.0
     img = np.concatenate([img, diff], axis=-1)
     img = np.expand_dims(img, axis=0)
     return img
time_on = 0
try:
    while True:
        img, pre = take_pic()
        diff = find_diff(img)
        img = normalize(img, diff)
        prediction = water_inference(img)
        print(f"\n prediction: {prediction}")
        if prediction > 0.6:
            f_prediction = food_inference(img)
            p_prediction = poop_inference(img)
            if f_prediction >0.7 or p_prediction > 0.7:
                print("predicted dirty")
            else:
                print('water predicted clean')
                time_on += 1
                with open("time.txt", 'w') as f:
                    f.write(str(time_on))
                    save_img(pre, prediction, diff)
        time.sleep(1)
finally:
    picam2.close()
    neo.clear_strip()
    neo.update_strip()
    print("cleanly shut down")
