import time
import numpy as np
import sys
import os
import json
import random
import tensorflow as tf
from PIL import Image

model = "/Users/jakehopkins/Downloads/if_water/clean_dirty.tflite"
file = "logs.json"
test= [3,2,1]
order = ["clean", "food", "no_water", "toilet"]
timeon = random.randint(10, 120)

def prediciton():
    interpreter = tf.lite.Interpreter(model_path=model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img_path = "/Users/jakehopkins/Downloads/if_water/Clean_Dirty/train/toilet/img_20260125_112105_7.jpg"
    
    # Load and preprocess image
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224))  # Adjust to your model's input size
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    interpreter.set_tensor(input_details[0]["index"], img_array) #sets up model

    interpreter.invoke() #runs model
    prediction = interpreter.get_tensor(output_details[0]["index"])
    # prediction = np.max(prediction)  # Get the index of the highest probability
    prediction = [round(x, 2) for x in prediction[0]]

    prediciton = np.argmax(prediction)
    prediciton = order[prediciton]
    print (order)
    print(prediction)
    return prediction
def log():
    # gets old data and adds it to list
    try:
        with open(file, "r") as f:
            old = json.load(f)
            ID = [e["ID"] for e in old if isinstance(e, dict) and "ID" in e]
            ID = max(ID) + 1 if ID else 1
    except(json.JSONDecodeError):
        old = []
        ID = 1

# edge case for empty file
    if not isinstance(old, list): 
        old = test

    old.append({ # adds new data onto end of old data list
        "water_status": prediciton,
        "time_on": str(timeon),
        "timestamp": str(time.strftime("%Y%m%d-%H%M%S")),
        "ID": ID
    }) 
    # adds appanded list to file
    with open(file, "w") as f:
        json.dump(old, f, indent=3)

prediciton()
log()