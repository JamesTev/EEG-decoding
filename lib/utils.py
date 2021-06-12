
def delay_ms(t):
    import utime
    t0 = utime.time()
    while (utime.time()-t0)*1000 < t:
        pass # TODO: investigate if this will actually free core during delay

def connect_wifi():
    import network
    import binascii

    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    if not wlan.isconnected():
        print('connecting to network...')
        wlan.connect('Teversham 2.4GHz', '083655655000')
        while not wlan.isconnected():
            pass
    print('network config:', wlan.ifconfig())
    return wlan

