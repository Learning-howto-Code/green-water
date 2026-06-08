# Waste Water Classification — A Deep Dive into My Sewer

# Final Product!
<img width="3264" height="2448" alt="IMG_1111" src="https://github.com/user-attachments/assets/6771587a-dcb3-4c84-aaab-6c318507442e" />

So, this project is pretty unique, so I will do my best to explain what exactly I'm trying to do...
I noticed that a lot of water goes down the drain in my house (duh), but I was curious what that water looked like, and where it came from.
I wanted to know what amount of the water my family used was greywater (acceptably clean, and usable for a second use), or blackwater (disgusting water from the toilet).
In order to do this I'm going to cut into my sewer (with my parents' permission), and install a camera/3D printed assembly to aid in the collection of data of what typical sewer water looks like. Once I have an understanding of it, I will train an array of ML models to classify, sort, and analyze the water to do things like detect where it came from, how much of it there is, and learn use patterns. I'm planning on the detection models using a CNN architecture, though I may also use an autoencoder and SVM for anomaly detection.

### 3D Model
<img width="1392" height="882" alt="Screenshot 2026-03-10 at 8 36 05 AM" src="https://github.com/user-attachments/assets/12b0c98c-9c46-426d-b728-1493936c63df" />


### Wiring Diagram
<img width="596" height="528" alt="Screenshot 2026-04-16 at 6 52 08 PM" src="https://github.com/user-attachments/assets/506af464-c054-4bcb-b8c4-4edac001cc8b" />
yes I just drew this on my iPad. KISS, keep it simple stupid

### BOM
| Name | Purpose | Qty | Total Cost (USD) | Distributor |
|------|---------|-----|-----------------|-------------|
| [3" Slip Fitting](https://www.grainger.com/product/4P009?gucid=N:N:PS:Paid:GGL:CSM-2295:UU5CX9:20800606:APZ_1&gclsrc=aw.ds&gad_source=1&gad_campaignid=22475795893&gclid=Cj0KCQjwgr_NBhDFARIsAHiUWr7TcDMXAryxJi2VdVsIToIjRWHAWOxhZWQ6h9fFvR5YjGmCXsIaxiQaArp3EALw_wcB) | To connect the T fitting to the rest of the pipe | 2 | $20.00 | Grainger |
| [3" ABS T Fitting](https://www.homedepot.com/pep/Charlotte-Pipe-3-in-ABS-DWV-Hub-x-Hub-x-Hub-2-Way-Cleanout-ABS004480600HD/313834553?source=shoppingads&locale=en-US&fp=ggl&pla=&mtc=SHOPPING-BF-CDP-GGL-D26P-026_001_PIPE_FITTING-NA-NA-NA-PMAX-NA-NA-NA-NA-NBR-NA-NA-NEW-_PMAXTEST&cm_mmc=SHOPPING-BF-CDP-GGL-D26P-026_001_PIPE_FITTING-NA-NA-NA-PMAX-NA-NA-NA-NA-NBR-NA-NA-NEW-_PMAXTEST-17697557984--&gclsrc=aw.ds&gad_source=1&gad_campaignid=17687574624&gbraid=0AAAAADq61UeUXh8CD3v1EYXGOttrxJvrZ&gclid=Cj0KCQjwgr_NBhDFARIsAHiUWr5VLB7fSKoUHgdbWhzxioN93eym2dgWzO-CDZ8OZz-KkbEUsmIKXGoaAiycEALw_wcB) | To allow the camera to see the water | 1 | $35.00 | Home Depot |
| [NeoPixel Ring x24](https://www.adafruit.com/product/1586?srsltid=AfmBOorzyPBVbfCt3g8E_ss7CaTj8lrPUNuM9SgI_s6fxNRJbnADYYTG) | Adjustable light for the camera | 1 | $17.00 | Adafruit |
| [Camera Module 2](https://www.adafruit.com/product/3099?src=raspberrypi) | The RPi camera for data collection | 1 | $30.00 | Adafruit |
| [Raspberry Pi 5 (4GB)](https://www.canakit.com/raspberry-pi-5-4gb.html?srsltid=AfmBOopFRKu7etXnwxGA9C9gw1PBtyJ5pvfmXzlo4iV10Dr-CeBKjEMu) | Edge device to compute the models | 1 | $85.00 | CanaKit |

**Total: $187.00**

> Note: I have the materials and am not requesting funding for my BOM.

