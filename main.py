import utime
from machine import Pin

led = Pin(5, Pin.OUT)

for i in range(10):
    led.value(not led.value())
    utime.sleep(0.1)

led.on() # logic inverted for built in led

def connect_wifi():
    import network

    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    t = utime.time()
    if not wlan.isconnected():
        print('connecting to network...')
        wlan.connect('Teversham 2.4GHz', '083655655000')
        while not wlan.isconnected():
            print('.')
            utime.sleep(0.25)
            
            if (utime.time()-t)>5:
                print('Network attempt timed out. ')
                break
            pass
    
    if wlan.isconnected():
        print('network config:', wlan.ifconfig())

# connect_wifi()

