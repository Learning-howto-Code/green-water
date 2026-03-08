import datetime
import time
import numpy as np
import cv2 as cv
import json
from tensorflow import keras
from PIL import Image
# from picamera2 import Picamera2
 


food_clean_model = "food_clean_aug.keras"
filepath= "/Users/jakehopkins/Downloads/if_water/food_clean/train/food/ID#1, 2026-03-07-08-15-34-282.jpg"

def food_clean():
    global img, img_array,frame, pred, old_file
    img = cv.imread(filepath)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    img = cv.resize(img, (224, 224))



    diff = 2
    
    # Load and preprocess image
    diff = np.full((224,224,), diff, dtype=np.float32)
    # Add diff as an additional channel
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.dstack((img, diff))
    img_array = np.expand_dims(img_array, axis=0)

    model = keras.models.load_model(food_clean_model)
    prediction = model.predict(img_array, verbose=0)
    pred_int = prediction
    pred_int = np.round(pred_int, 4) #rounds pred_int to 4 points
    prediction = [round(x, 2) for x in prediction[0]]
    food_pred_int = prediction
    if pred_int > 0.5:
        food_pred = "clean"
    if pred_int <= 0.5:
        food_pred = "dirty"
    return food_pred, food_pred_int, prediction
current_pred = None
start = time.time()

prediction=food_clean()
print(prediction)