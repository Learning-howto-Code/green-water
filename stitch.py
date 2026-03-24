# Source - https://stackoverflow.com/a/44948030
# Posted by BoboDarph, modified by community. See post 'Timeline' for change history
# Retrieved 2026-03-17, License - CC BY-SA 4.0

import cv2
import os

image_folder = 'images'
video_name = 'video.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 1, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()
