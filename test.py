import re
import os
from datetime import datetime

def chronological_key(filename):
    # Supports both patterns:
    # 1) img_20260125_114009_1247.jpg
    # 2) ID#1, 2026-03-19-17-59-23-334.jpg
    m1 = re.search(r"(\d{8})_(\d{6})_(\d+)", filename)
    if m1:
        date_str, time_str, ms_str = m1.groups()
        dt = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")
        # Normalize to microseconds (max 6 digits).
        us = int(ms_str[:6].ljust(6, "0"))
        return dt.replace(microsecond=us)

    m2 = re.search(r"(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d+)", filename)
    if m2:
        y, mo, d, h, mi, s, ms_str = m2.groups()
        us = int(ms_str[:6].ljust(6, "0"))
        return datetime(int(y), int(mo), int(d), int(h), int(mi), int(s), us)

# Generate sorted filelist for ffmpeg

folder = "/Users/jakehopkins/Downloads/if_water/movie"

images = [f for f in os.listdir(folder) if f.endswith(('.jpg', '.png'))]
with open("filelist.txt", "w") as f:
    for img in images:
        f.write(f"file '{os.path.join(folder, img)}'\n")

# Then run ffmpeg
os.system("ffmpeg -f concat -safe 0 -i filelist.txt -r 30 -c:v libx264 -pix_fmt yuv420p video.mp4")