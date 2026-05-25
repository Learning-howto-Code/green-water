import cv2 as cv
import numpy as np
import time
from datetime import datetime
import tensorflow as tf
tflite = tf.lite

def water_pred():
    preds = []
    for i in range(10):
        user = float(input("enter water pred"))
        preds.append(user) #emulates model inference
        print(np.average(preds))
        time.sleep(1/5)
        if i > 4:
            preds = preds[1:] # removes first value to keep avg to last 5 preds
        if preds is not None and np.average(preds) > .6:
            prescence = True
        else:
            prescence = False
        print(prescence)  
def load_model(model_path):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()        
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return {
        "interpreter": interpreter,
        "input_details": input_details,
        "output_details": output_details
    }
img = '/Users/jakehopkins/Downloads/if_water/pi_data/val/water/img_20251231_170938_14.jpg'
img = cv.imread(img)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img = cv.resize(img, (224, 224))
img = img.astype("float32") / 255.0
img = np.expand_dims(img, axis=0)
def water_inference(model_path, diff, diff_map): 
    global img
    model = load_model(model_path)
    diff = False
    interpreter = model["interpreter"]
    input_details = model["input_details"]
    output_details = model["output_details"]
    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()
    water_prediction = interpreter.get_tensor(output_details[0]["index"])
    print("water_prediction:", water_prediction, "------------------", end="\n\n")
    return water_prediction

load_model('models/if_water.tflite')


preds = []
for i in range(10):
    water_prediction = water_inference('models/if_water.tflite', False, False)
    preds.append(water_prediction) #emulates model inference
    print(np.average(preds))
    time.sleep(1/5)
    if i > 4:
        preds = preds[1:] # removes first value to keep avg to last 5 preds
    if preds is not None and np.average(preds) > .6:
        prescence = True
    else:
        prescence = False
    print(prescence)  