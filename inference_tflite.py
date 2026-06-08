import numpy as np
import cv2 as cv
from tensorflow.lite.python import interpreter as tflite
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

img_dir = '/Users/jakehopkins/Downloads/6-4'
true_label = "water"  # set to "water" or "no water"

if_water_model = '/Users/jakehopkins/Downloads/if_water/models/if_water.tflite'
interpreter = tflite.Interpreter(model_path=if_water_model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input dtype:", input_details[0]['dtype'])
print("Input shape:", input_details[0]['shape'])
input_dtype = input_details[0]['dtype']

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
image_files = [
    f for f in os.listdir(img_dir)
    if os.path.splitext(f)[1].lower() in VALID_EXTS
]
 
if not image_files:
    print(f"No images found in {img_dir}")
    exit(1)

y_true = []
y_pred = []

for fname in image_files:
    path = os.path.join(img_dir, fname)
    img = cv.imread(path)
    if img is None:
        print(f"Skipping (unreadable): {fname}")
        continue

    img = cv.resize(img, (224, 224))
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    if input_dtype == np.uint8:
        img_array = np.expand_dims(np.array(img, dtype=np.uint8), axis=0)
    else:
        img_array = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)

    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]["index"])
    score = float(prediction[0][0]) if prediction[0].shape else float(prediction[0])

    pred_label = "water" if score > 0.5 else "no water"
    print(f"{fname}: {score:.4f} → {pred_label}")

    y_true.append(true_label)
    y_pred.append(pred_label)

labels = ["water", "no water"]
cm = confusion_matrix(y_true, y_pred, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap="Blues")
plt.title("if_water Confusion Matrix")
plt.tight_layout()
plt.show()
