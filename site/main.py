from flask import Flask, flash, render_template, render_template_string, request, Response
# from picamera2 import Picamera2
import cv2 as cv
import os
import time
import datetime

app = Flask(__name__)
app.secret_key = '1'

done = True
def toilet_capture():
    global done
    done = False
    folder_time = time.strftime("%m %d %H:%M")
    os.mkdir(f"/Users/jakehopkins/Downloads/if_water/data/toilet{folder_time}")
    for i in range(100*30): #captures frames at 30fps for 100 seconds
        frame = picam2.capture_array()
        img = cv.cvtColor(frame, cv.COLOR_BGR2RGB) #the pi cam takes in BGR
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + f"-{datetime.datetime.now().microsecond // 1000:03d}"
        img_name = f"{timestamp}.jpg"
        cv.imwrite(img_name, img)
    done = True
def sink_capture():
    global done
    done = False
    folder_time = time.strftime("%m%d%H:%M")
    base_dir="/Users/jakehopkins/Downloads/if_water/data/sink"
    out_dir = os.path.join(base_dir, folder_time)
    os.makedirs(out_dir, exist_ok=True)
    img = "/Users/jakehopkins/Downloads/if_water/food_clean_extra_clean/01442_img_20260124_130809_1446.jpg"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + f"-{datetime.datetime.now().microsecond // 1000:03d}"
    img_name = f"{timestamp}.jpg"
    cv.imwrite(os.path.join(out_dir, img_name), img)
    done = True

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        key = request.form.get("key")
        if key == "wgb":
            if done == True:
                toilet_capture()
                flash("Capturing images")
            else:
                flash("Capture in progress, wait to hit it again") 
        if key == "t":
            if done == True:
                sink_capture()
                flash("Capturing images")
            else:
                flash("Capture in progress, wait to hit it again") 
    return render_template("main.html")



# # Picamera2 setup
# picam2 = Picamera2()
# picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
# picam2.start()

# def generate_frames():
#     while True:
#         frame = picam2.capture_array()
#         ret, buffer = cv2.imencode('.jpg', frame)
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
    app.run(host='0.0.0.0', port=8000, debug=True)