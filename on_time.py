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
    
    total_seconds += 1
    time.sleep(1)

