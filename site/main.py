from flask import Flask, flash, render_template, render_template_string, request, Response
from picamera2 import Picamera2
from pi5neo import Pi5Neo
import cv2 as cv
import os
import time
import datetime



app = Flask(__name__)
app.secret_key = '1'

# Picamera2 setup
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()

SPI_DEVICE = '/dev/spidev0.0' # Rpi protocol to get the timing right for the GPIOs
SPI_SPEED_KHZ = 800 #speed of SPI protocol

neo = Pi5Neo(SPI_DEVICE, 30, SPI_SPEED_KHZ)

done = True
def toilet_capture():
    global done
    done = False

    neo.fill_strip(220, 240, 120)
    neo.update_strip()  # commit/send to LEDs
    time.sleep(0.5)

    folder_time = time.strftime("%m:%d_%H:%M")
    base_dir="/home/jake/Downloads/if-water-cnn/data/toilet"
    out_dir = os.path.join(base_dir, folder_time)
    os.makedirs(out_dir, exist_ok=True)
    for i in range(100*30):
        frame = picam2.capture_array()
        img = cv.cvtColor(frame, cv.COLOR_BGR2RGB) #the pi cam takes in BGR
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + f"-{datetime.datetime.now().microsecond // 1000:03d}"
        img_name = f"{timestamp}.jpg"
        cv.imwrite(os.path.join(out_dir, img_name), img)
    done = True
def sink_capture():
    global done
    done = False

    neo.fill_strip(220, 240, 120)
    neo.update_strip()  # commit/send to LEDs
    time.sleep(0.5)
    
    folder_time = time.strftime("%m:%d_%H:%M")
    base_dir="/home/jake/Downloads/if-water-cnn/data/sink"
    out_dir = os.path.join(base_dir, folder_time)
    os.makedirs(out_dir, exist_ok=True)
    for i in range(200*30):
        frame = picam2.capture_array()
        img = cv.cvtColor(frame, cv.COLOR_BGR2RGB) #the pi cam takes in BGR
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + f"-{datetime.datetime.now().microsecond // 1000:03d}"
        img_name = f"{timestamp}.jpg"
        cv.imwrite(os.path.join(out_dir, img_name), img)
    done = True

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        key = request.form.get("key")
        if key == "gb":
            if done == True:
                flash("Capturing images")
                toilet_capture()
            else:
                flash("Capture in progress, wait to hit it again") 
        if key == "t":
            if done == True:
                flash("Capturing images")
                sink_capture()
            else:
                flash("Capture in progress, wait to hit it again") 
    return render_template("main.html")





# def generate_frames():
#     while True:
#         frame = picam2.capture_array()
#         ret, buffer = cv.imencode('.jpg', frame)
#         if not ret:
#             continue  # skip frame if encoding fails
#         jpg_bytes = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + jpg_bytes + b'\r\n')

# @app.route('/stream.mjpg')
# def stream():
#     return Response(generate_frames(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True, use_reloader=False)