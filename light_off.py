import board
import neopixel_spi as neopixel
from time import sleep
import sys

#Pins 5v=2, GND=6, DIN=19

pixels = neopixel.NeoPixel_SPI(
    board.SPI(), 24, brightness=1.0, auto_write=False, pixel_order=neopixel.GRB
)

pixels.fill((255, 255, 255))
pixels.show()  # commit/send to LEDs
sleep(1)
pixels.fill((0, 0, 0))
pixels.show()



