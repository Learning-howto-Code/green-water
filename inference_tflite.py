import datetime
import time
import numpy as np
import cv2 as cv
import json
import tflite_runtime.interpreter as tflite
from PIL import Image
import os


seconds = int(20) #how long to run the script for.

if_water_model = "if_water.tflite"
food_clean_model = "food_clean.tflite"
file = "logs.json"

img_path = "ID#1, 2026-03-02-16-31-49-197.jpg"

def if_water():
    global img, img_array,frame, pred, pred_int, img_path

    img = cv.imread(img_path)
    img = cv.resize(img, (224, 224))

    interpreter = tflite.Interpreter(model_path=if_water_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)
    
    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]["index"])
    pred_int = prediction
    pred_int = np.round(pred_int, 4) #rounds pred_int to 4 points
    prediction = [round(x, 2) for x in prediction[0]]
    if pred_int > 0.5:
        pred = "water"
    if pred_int <= 0.5:
        pred = "no water"
    return prediction, 
old_file = None
current_pred = None
start = time.time()
def food_clean():
    global img, img_array,frame, pred, old_file, img_path
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {os.path.abspath(img_path)}")

    img = cv.imread(img_path)
    img = cv.resize(img, (224, 224))

    if old_file is None:
        old_file = img
    new_file = img

    #diff calculation
    # old = cv.imread(old_file )     
    #new_img = cv.imread(new_file)
    diff = cv.absdiff(old_file, new_file)
    diff = np.average(diff)
    old_file = new_file
    print(f"diff is {diff} out 255")
    

    interpreter = tflite.Interpreter(model_path=food_clean_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Load and preprocess image
    diff = np.full((224,224,), diff, dtype=np.float32)
  # Add diff as an additional channel
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.dstack((img, diff))
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]["index"])
    pred_int = prediction
    pred_int = np.round(pred_int, 4) #rounds pred_int to 4 points
    prediction = [round(x, 2) for x in prediction[0]]
    food_pred_int = prediction
    if pred_int > 0.5:
        food_pred = "clean"
    if pred_int <= 0.5:
        food_pred = "dirty"
    return food_pred_int
current_pred = None
start = time.time()



previous_prediction = None
food_number=food_clean()
water_number=if_water()

