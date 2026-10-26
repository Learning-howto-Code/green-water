from picamera2 import Picamera2
from time import sleep, strftime, time
import board
import neopixel_spi as neopixel
import os
import cv2
from tqdm import tqdm
# Image folder
img_folder = "/home/jake/Downloads/if_water/data/Clean_Dirty/no_water"

Frames=3500
brightness=1

# SPI setup for NeoPixel
pixels = neopixel.NeoPixel_SPI(
    board.SPI(), 30, brightness=brightness, auto_write=False, pixel_order=neopixel.GRB
)
 

def take_pic():
    pixels.fill((220, 240, 120)) #sets LED's to white and a little dimmer
    pixels.show() #sets color
    print("light on")
    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"size": (640, 360)}, lores={"size": (640, 360)}, display="lores")
    picam2.configure(config) #sets configuration
    picam2.start()
    print("activated cam")
    sleep(2)    #watis for cam to start
    print("waited 2 sec") #cam is now ready
    frame_count=0
    start_time = time()
    for frame_count in tqdm(range(1,Frames+1),desc="Capturing", unit="img"):  #takes images for 20 seconds
        frame = picam2.capture_array()  #uses capture array funtion instead
        frame= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #the pi cam takes in BGR
        filename = f"{img_folder}/img_{strftime('%Y%m%d_%H%M%S')}_{frame_count}.jpg"
        cv2.imwrite(filename, frame)

    print("Captured", Frames, "frames in", round(time()-start_time, 2), "seconds")
    picam2.stop()
    picam2.close()


take_pic()
pixels.fill((0, 0, 0)) #turns LED's off
pixels.show() #sets color
