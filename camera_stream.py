from picamera2 import Picamera2 # type: ignore
from time import sleep
import time
from datetime import date
import numpy as np
import cv2
import tflite_runtime.interpreter as tflite # type: ignore
import sys
import os
import pi5neo as Pi5Neo # type: ignore
import json
import subprocess
import psutil

picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (224, 224)}, buffer_count=4)
picam2.configure(config)
picam2.start()

