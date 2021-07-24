import utime
from machine import Pin
import gc

led = Pin(26, Pin.OUT)
led_red = Pin(13, Pin.OUT)
btn = Pin(32, Pin.IN, Pin.PULL_UP)

def flash(iterations, duration):
    for i in range(iterations):
        led.value(not led.value())
        utime.sleep(duration)
        
flash(5, 0.1)

t0 = utime.time()
run_main = False

led_red.value(0)
led.value(1)

while utime.time()-t0 < 2.5 or run_main:
    if btn.value()==0:
        # run the main process
        led_red.value(1)
        run_main = True
        from lib.core import initialise, run
        gc.collect()
        initialise()
        gc.collect()
        run()
        
# exit and continue to normal dev mode
flash(2, 0.2)
led.value(0)


