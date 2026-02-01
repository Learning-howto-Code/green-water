input_path = "/Users/jakehopkins/Downloads/production_water.MOV"
output_dir = "/Users/jakehopkins/Downloads/if_water/production/water"

import cv2
import os

def extract_frames(input_path, output_dir):
    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {input_path}")

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # No more frames

        # Build filename: frame_00001.jpg
        filename = f"frame_{frame_idx:05d}.jpg"
        filepath = os.path.join(output_dir, filename)

        # Save the frame
        cv2.imwrite(filepath, frame)

        frame_idx += 1

    cap.release()
    print(f"Done. Extracted {frame_idx} frames to {output_dir}")




extract_frames(input_path, output_dir)
