
def delay_ms(t):
    import utime
    t0 = utime.time()
    while (utime.time()-t0)*1000 < t:
        pass # TODO: investigate if this will actually free core during delay
    
def update_buffer(buf, el, max_size):
    if type(el) in [float, int]:
        el = [el]
    el = el[-max_size:]
    return (buf[-(max_size-len(el)):]+el)[-max_size:]

def connect_wifi(ssid, password):
    import network
    import binascii

    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    if not wlan.isconnected():
        print('connecting to network...')
        wlan.connect(ssid, password)
        while not wlan.isconnected(): # okay that this is blocking
            pass
    print('network config:', wlan.ifconfig())
    return wlan

def load_env_vars(path):
    import ure as re

    envre = re.compile(r'''^([^\s=]+)=(?:[\s"']*)(.+?)(?:[\s"']*)$''')
    result = {}
    with open(path) as ins:
        for line in ins:
            match = envre.match(line)
            if match is not None:
                result[match.group(1)] = match.group(2).replace('\n', '')
    return result

