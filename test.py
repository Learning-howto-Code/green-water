from picamera2 import Picamera2 # type: ignore
from time import sleep
import time
from datetime import date, datetime
import numpy as np
import cv2 as cv
import tflite_runtime.interpreter as tflite # type: ignore
import os
import pi5neo  # type: ignore
from collections import deque
import threading
import queue
#vars
dir = f"data/{date.today()}"
model = "models/if_water.tflite"
writeq = queue.Queue()
ring = deque(maxlen=60)   # last 60 frames, jpg-encoded
#turns on light
SPI_DEVICE = '/dev/spidev0.0' # Rpi protocol to get the timing right for the GPIOs
SPI_SPEED_KHZ = 800 #speed of SPI protocol

neo = pi5neo.Pi5Neo(SPI_DEVICE, 30, SPI_SPEED_KHZ) #Pins 5v=2, GND=6, DIN=19

neo.fill_strip(220, 240, 120)
neo.update_strip()  # commit/send to LEDs
time.sleep(1)
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
    save_img(img, prediction)
    return prediction
def save_img(img, prediction):
    os.makedirs(dir, exist_ok=True)
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]   # ms
    filename = f"{dir}/{time} | {prediction}.jpg"
    cv.imwrite(filename, img)
def rectangle(prediction):
    if prediction < 0.5: # means guesses water
            color= (0, 255, 0) # green
    else:
        color = (0, 0, 255) # red
    h = pre.shape[0]
    cv.rectangle(pre, (0, 0), (50, 50), color, -1)
    save_img(pre, prediction)
    print("taking pic")
def lookback(prediction, change):
    ok, buf = cv.imencode(".jpg", pre)   # encode once, keep bytes not the array
    ring.append((datetime.now(), float(np.asarray(prediction).flat[0]), buf.tobytes()))
    if change:
        writeq.put(list(ring))   # snapshot: ring keeps rolling while writer works
def write_lookback():   # runs in its own thread, drains writeq until sentinel
    while True:
        snapshot = writeq.get()
        if snapshot is None:
            break
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]   # ms
        sub_dir = os.path.join(dir, stamp)
        os.makedirs(sub_dir, exist_ok=True)
        for i, (ts, pred, jpg) in enumerate(snapshot):
            name = f"{i:03d}_{ts.strftime('%H-%M-%S-%f')[:-3]}_p{pred:.3f}.jpg"
            with open(os.path.join(sub_dir, name), "wb") as f:
                f.write(jpg)
        print(f"wrote {len(snapshot)} frames -> {sub_dir}")

old = None
writer = threading.Thread(target=write_lookback)
writer.start()
try:
    while True:
        img = take_pic()
        prediction = water_inference(img)
        rectangle(prediction)
        if prediction < 0.5:
            state = "no_water"
        else:
            state = "water"
        if state != old:
            change = True
        else:
            change = False
        old = state
        lookback(prediction, change)
        time.sleep(5)
finally:
    writeq.put(None)   # tell writer to finish
    writer.join(timeout=10)
    picam2.close()
    neo.clear_strip()
    neo.update_strip()
    print("\n cleanly shut down")
