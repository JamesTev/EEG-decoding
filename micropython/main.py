from machine import Pin
import gc

from lib.core import initialise, run

gc.collect() # free up any memory used in imports

initialise()
gc.collect()
run()