import subprocess


def is_ssh_active() -> bool: # returns true if ssh is active
    result = subprocess.run(["who"], capture_output=True, text=True, check=True)
    for line in result.stdout.splitlines(): # if loop for each line of output
        if "(" in line and ")" in line: #checks if line has ssh indicators
            return True # returns true if it finds indicators
    return False# else returns false
if_ssh = is_ssh_active()
if if_ssh == False:
    from pi5neo import Pi5Neo

    spi_channel = "/dev/spidev0.0"
    spi_hz = 800
    leds = 24

    neo = Pi5Neo(spi_channel, leds, spi_hz)
    neo.fill_strip(0, 0, 0)
    neo.update_strip()
    print("SSH not active, turning off LED's")
else :
    print("SSH active, not touchng LED's")
