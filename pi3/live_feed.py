#live camera feed in the browser, open http://<pi-ip>:8001 to view
#based on the picamera2 mjpeg_server example
#gets frames from send_img.py so other scripts can use the camera at the same time
import io
import socketserver
import threading
import time
from http import server
from threading import Condition
import zmq

PORT = 8001
CAMERA = "tcp://localhost:5555" #send_img.py
FPS = 10

PAGE = """\
<html>
<head><title>Pi Camera</title></head>
<body style="margin:0;background:#111">
<img src="stream.mjpg" style="width:100%;height:auto">
</body>
</html>
"""

class StreamingOutput(io.BufferedIOBase): # holds the newest jpeg frame
    def __init__(self):
        self.frame = None
        self.condition = Condition()
        self.viewers = 0 # open browser tabs, only grab frames when someone is watching

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()

class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            content = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.end_headers()
            with output.condition:
                output.viewers += 1
            try:
                while True: # sends each new frame to the browser
                    with output.condition:
                        output.condition.wait()
                        frame = output.frame
                    self.wfile.write(b"--FRAME\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError): # browser tab closed
                pass
            finally:
                with output.condition:
                    output.viewers -= 1
        else:
            self.send_error(404)

class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

def connect(ctx):
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 3000) # don't hang forever if send_img.py is down
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(CAMERA)
    return sock

def fetch_frames(): # asks send_img.py for frames while someone is watching
    ctx = zmq.Context()
    sock = connect(ctx)
    while True:
        if output.viewers == 0:
            time.sleep(0.5)
            continue
        try:
            sock.send_string("image")
            jpg, stamp = sock.recv_multipart()
            output.write(jpg)
        except zmq.Again: # no reply, REQ socket is stuck now so make a new one
            print("no reply from send_img.py, retrying")
            sock.close()
            sock = connect(ctx)
        time.sleep(1 / FPS)

output = StreamingOutput()
threading.Thread(target=fetch_frames, daemon=True).start()

try:
    print(f"streaming on http://<pi-ip>:{PORT}")
    StreamingServer(("", PORT), StreamingHandler).serve_forever()
except KeyboardInterrupt:
    pass
finally:
    print("cleanly shut down")
