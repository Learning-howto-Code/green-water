#RunAWaterModelConstantly
#logs every second there's water
#
import tflite_runtime.interpreter as tflite # type: ignore
import time
import datetime
import cv2 
from picamera2 import Picamera2 # type: ignore
import pi5neo # type: ignore
import numpy as np
import psutil
import subprocess

model = "models/if_water.tflite"
lookback = 5
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

while True:
    img = take_pic()
    water_prediction = run_model(img)
    pred_list.append(water_prediction)

    if len(pred_list) > lookback:
        pred_list = pred_list[1:]
    water_prediction = np.average(pred_list)

    if water_prediction > 0.6:
        print(f"water detected: {str(water_prediction)}")
        seconds_on += 1
    else:
        print(f"no water detected: {str(water_prediction)}")

    if seconds_on % 50 == 0:
        with open("log.txt", "w") as f:
            f.write(f"Seconds on: {str(seconds_on)} out of {str(total_seconds)} seconds\n")
    
    if total_seconds % 10 == 0:
        hardware_data()
    total_seconds += 1
    time.sleep(1)

