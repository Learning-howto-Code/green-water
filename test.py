import threading
import time
import sys
import tty
import termios

def function1():
    while True:
        print("running", end="\r\n")
        time.sleep(1)

def listener():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch == 'x':
                print("\r\ntaking picture", end="\r\n")
            elif ch == '\x03':  # ctrl+c to quit
                sys.exit(0)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

t = threading.Thread(target=listener, daemon=True)
t.start()
function1()
