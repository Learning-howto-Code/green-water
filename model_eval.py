import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import *
from keras.models import *
from datetime import datetime
from keras.preprocessing import image
from sklearn.metrics import classification_report, classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
    # Extract true labels and predictions from tf.data.Dataset
    y_true = []
    y_pred_probs = []
    
    for images, labels in test_data:
        y_true.extend(labels.numpy())
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