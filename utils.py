import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import *
from keras.models import *
from datetime import datetime
from keras.preprocessing import image
from sklearn.metrics import confusion_matrix, precision_recall_curve
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
import os
import cv2 as cv
import numpy as np

def plot(history, timestamp):
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='train')
    plt.plot(val_acc, label='val')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='train')
    plt.plot(val_loss, label='val')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()
    plt.close()


# Confusion matrix on test set
def matrix(model, test_data):

    # Extract true labels and predictions from ImageDataGenerator
    y_true = []
    y_pred_probs = []
    
    for images, labels in test_data:
        y_true.extend(labels)  # Remove .numpy()
        predictions = model.predict(images, verbose=0)
        y_pred_probs.extend(predictions)
    
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    
    cm = confusion_matrix(y_true, y_pred)
    print("\nCONFUSION MATRIX")
    print(cm)
    
    print(f"\nTrue Positives: {cm[1][1]}")
    print(f"False Positives: {cm[0][1]}")
    print(f"False Negatives: {cm[1][0]}")
    print(f"True Negatives: {cm[0][0]}")
    
    # Calculate accuracy
    accuracy = (cm[0][0] + cm[1][1]) / cm.sum()
    print(f"\nAccuracy: {accuracy:.4f}")

def precision_recall(model, test_data):
    title="Precision-Recall Curve"
    y_true = []
    y_pred_probs = []

    for images, labels in test_data:
        y_true.extend(labels)
        predictions = model.predict(images, verbose=0)
        y_pred_probs.extend(predictions)

    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs).flatten()

    precision, recall, _ = precision_recall_curve(y_true, y_pred_probs)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label="PR curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()
def diffs():

    

    global new_filepath, diff
    old_data = []
    if os.path.exists("delta.json"):
        with open("delta.json", "r") as f:
            try:
                loaded = json.load(f)
                old_data = loaded if isinstance(loaded, list) else [] # test case for empty file
            except json.JSONDecodeError:
                old_data = []
    for dirpath in os.walk("/Users/jakehopkins/Downloads/if_water/food_clean"):
        dir = dirpath[0]
        print(dir)
        files = [f for f in os.listdir(dir) if f.endswith('.jpg')]
        old_filepath = "/Users/jakehopkins/Downloads/if_water/food_clean/test/clean/img_20260125_113927_2.jpg"
        for i in range(len(files)):
            files = [f for f in os.listdir(dir) if f.endswith('.jpg')]
            files = sorted(files)  # sorts the files by timetamp TIMESTAMPS SKIP A FEW NUMBERS
            new_filepath = os.path.join(dir, files[i])

            
            old = cv.imread(old_filepath)
            
            new = cv.imread(new_filepath)
        
            diff = cv.absdiff(old, new)
            diff = np.average(diff)

            old_filepath = new_filepath
            print(f"new filepath {new_filepath}")
            print(f"old filepath {old_filepath}")
            print(f"loop number {i}")
            print(f"difference is {diff} out 255")            
            old_data.append({
                "filepath": new_filepath,
                "diff": float(diff)
            })
    with open("delta.json", "w") as f:
        json.dump(old_data, f, indent=2)   

def prod_plot():
    with open("logs.json") as f:
        data = json.load(f)
    
    timestamps = []
    water_states = []
    times_on = []
    
    for entry in data:
        if isinstance(entry, dict) and "timestamp" in entry and "water_status" in entry:
            timestamps.append(entry["timestamp"])
            water_states.append(entry["water_status"])
            times_on.append(float(entry["time_on"]))
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Convert states to numeric values for plotting
    state_values = [1 if state == "water" else 0 for state in water_states]
    
    # Plot as step function
    plt.step(range(len(state_values)), state_values, where='mid', linewidth=2, marker='o')
    plt.xlabel('Observation')
    plt.ylabel('Water State')
    plt.title('Water State Changes Over Time')
    plt.yticks([0, 1], ['No Water', 'Water'])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.close()

