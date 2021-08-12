from machine import Pin
import gc
from lib.core import initialise, run
gc.collect()

led_red = Pin(13, Pin.OUT)

led_red.value(1)

initialise()
led_red.value(0)
gc.collect()
run()