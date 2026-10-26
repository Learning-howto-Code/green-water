import subprocess


def is_ssh_active() -> bool: # returns true if ssh is active
    result = subprocess.run(["who"], capture_output=True, text=True, check=True)
    for line in result.stdout.splitlines(): # if loop for each line of output
        if "(" in line and ")" in line: #checks if line has ssh indicators
            return True # returns true if it finds indicators
    return False# else returns false
if_ssh = is_ssh_active()
if if_ssh == False:
    import board
    import neopixel_spi as neopixel

    pixels = neopixel.NeoPixel_SPI(
        board.SPI(), 24, brightness=1.0, auto_write=False, pixel_order=neopixel.GRB
    )
    pixels.fill((0, 0, 0))
    pixels.show()
    print("SSH not active, turning off LED's")
else :
    print("SSH active, not touchng LED's")
