#live camera feed in the browser, open http://<pi-ip>:8000 to view
#based on the picamera2 mjpeg_server example
import io
import socketserver
from http import server
from threading import Condition
import board
import neopixel_spi as neopixel # type: ignore
from libcamera import Transform # type: ignore
from picamera2 import Picamera2 # type: ignore
from picamera2.encoders import JpegEncoder # type: ignore
from picamera2.outputs import FileOutput # type: ignore

PORT = 8000

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
        else:
            self.send_error(404)

class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

#turns on light so the feed matches what the model sees
pixels = neopixel.NeoPixel_SPI(
    board.SPI(), 24, brightness=1.0, auto_write=False, pixel_order=neopixel.GRB
)
pixels.fill((220, 240, 120))
pixels.show()

#hflip+vflip is the same as the 180 rotation in the other scripts
picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"size": (1280, 720)}, transform=Transform(hflip=1, vflip=1)
)
picam2.configure(config)
output = StreamingOutput()
picam2.start_recording(JpegEncoder(), FileOutput(output))

try:
    print(f"streaming on http://<pi-ip>:{PORT}")
    StreamingServer(("", PORT), StreamingHandler).serve_forever()
except KeyboardInterrupt:
    pass
finally:
    picam2.stop_recording()
    picam2.close()
    pixels.fill((0, 0, 0))
    pixels.show()
    print("cleanly shut down")
