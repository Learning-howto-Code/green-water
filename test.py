import datetime
import time
import numpy as np
import cv2 as cv
import json
import tflite_runtime.interpreter as tflite
from picamera2 import Picamera2
from pi5neo import Pi5Neo

# --- Config ---
SECONDS = 20
FPS = 30
IF_WATER_MODEL = "if_water.tflite"
FOOD_CLEAN_MODEL = "food_clean.tflite"
LOG_FILE = "logs.json"
IMG_DIR = "logged_data/both_models"

# --- Hardware setup ---
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (640, 360)}, buffer_count=4)
picam2.configure(config)
picam2.start()

SPI_DEVICE = '/dev/spidev0.0'
SPI_SPEED_KHZ = 800
neo = Pi5Neo(SPI_DEVICE, 30, SPI_SPEED_KHZ)
neo.fill_strip(220, 240, 120)
neo.update_strip()
time.sleep(0.5)

# --- Load interpreters once ---
water_interp = tflite.Interpreter(model_path=IF_WATER_MODEL)
water_interp.allocate_tensors()
water_input = water_interp.get_input_details()
water_output = water_interp.get_output_details()

food_interp = tflite.Interpreter(model_path=FOOD_CLEAN_MODEL)
food_interp.allocate_tensors()
food_input = food_interp.get_input_details()
food_output = food_interp.get_output_details()

# --- State ---
old_frame = None
current_pred = None
start_time = None
time_on = 0


def get_next_id():
    """Read logs.json and return next available ID."""
    try:
        with open(LOG_FILE, "r") as f:
            old = json.load(f)
            ids = [e["ID"] for e in old if isinstance(e, dict) and "ID" in e]
            return max(ids) + 1 if ids else 1
    except (json.JSONDecodeError, FileNotFoundError):
        return 1


def capture_frame(frame_id):
    """Capture a frame, save it, return the resized 224x224 image."""
    frame = picam2.capture_array()
    img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") \
        + f"-{datetime.datetime.now().microsecond // 1000:03d}"
    cv.imwrite(f"{IMG_DIR}/ID#{frame_id}, {timestamp}.jpg", img)
    return cv.resize(img, (224, 224))


def run_if_water(img):
    """Run if_water model. Returns (label, confidence)."""
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    water_interp.set_tensor(water_input[0]["index"], img_array)
    water_interp.invoke()
    pred = water_interp.get_tensor(water_output[0]["index"])
    conf = round(float(pred[0][0]), 4)
    label = "water" if conf > 0.5 else "no water"
    return label, conf


def run_food_clean(img, prev_frame):
    """Run food_clean model with diff channel. Returns (label, confidence, current_frame)."""
    if prev_frame is None:
        prev_frame = img

    diff_val = np.average(cv.absdiff(prev_frame, img))
    diff_channel = np.full((224, 224), diff_val, dtype=np.float32)

    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.dstack((img_array, diff_channel))  # 4 channels: RGB + diff
    img_array = np.expand_dims(img_array, axis=0)

    food_interp.set_tensor(food_input[0]["index"], img_array)
    food_interp.invoke()
    pred = food_interp.get_tensor(food_output[0]["index"])
    conf = round(float(pred[0][0]), 4)
    # folders: clean/ = 0, food/ = 1
    label = "food" if conf > 0.5 else "clean"
    return label, conf, img  # return img as next prev_frame


def log_prediction(water_label, food_label, food_conf, frame_id):
    """Log prediction to JSON when food state changes."""
    global current_pred, start_time, time_on

    now = time.time()

    if current_pred is None:
        current_pred = food_label
        start_time = now
        return

    if food_label != current_pred:
        time_on = round(now - start_time, 3)
        current_pred = food_label
        start_time = now

    try:
        with open(LOG_FILE, "r") as f:
            old = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        old = []

    if not isinstance(old, list):
        old = []

    old.append({
        "water_status": water_label,
        "food_status": food_label,
        "food_confidence": food_conf,
        "time_on": str(time_on),
        "timestamp": time.strftime("%Y%m%d-%H%M%S"),
        "ID": frame_id,
    })

    with open(LOG_FILE, "w") as f:
        json.dump(old, f, indent=3)


# --- Main loop ---
frame_id = get_next_id()
previous_food = None

for _ in range(SECONDS * FPS):
    img = capture_frame(frame_id)

    water_label, water_conf = run_if_water(img)
    food_label, food_conf, old_frame = run_food_clean(img, old_frame)

    print(f"{water_label} ({water_conf})  |  {food_label} ({food_conf})")

    if food_label != previous_food:
        log_prediction(water_label, food_label, food_conf, frame_id)
        previous_food = food_label

    frame_id += 1

# --- Cleanup ---
time.sleep(0.5)
neo.fill_strip(0, 0, 0)
neo.update_strip()
