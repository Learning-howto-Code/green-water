from picamera2 import Picamera2
from time import sleep, strftime, time
from pi5neo import Pi5Neo
import os
import cv2
from tqdm import tqdm
# Image folder
img_folder = "/home/jake/Downloads/if_water/Clean_Dirty/food"
Frames=3
brightness=1

# SPI setup for NeoPixel
SPI_DEVICE = '/dev/spidev0.0'
SPI_SPEED_KHZ = 800
pin = 30
neo = Pi5Neo(SPI_DEVICE, pin, SPI_SPEED_KHZ)
 
def take_pic():
    neo.fill_strip(255, 255, 255) #sets LED's to white and a little dimmer
    #neo.update_strip() #sets color
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
    
def take_pic2():
    neo.fill_strip(220*brightness, 240*brightness, 120*brightness) #sets LED's to white and a little dimmer
    neo.update_strip() #sets color
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
take_pic2()
neo.fill_strip(0, 0, 0) #sets LED's to white and a little dimmer
neo.update_strip() #sets color
