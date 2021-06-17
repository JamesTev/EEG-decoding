
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
        wlan.connect('Harry Wifi', '5BjkgMCp1nJK')
        while not wlan.isconnected():
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
                result[match.group(1)] = match.group(2).replace('\n', '').replace(' ', '')
    return result

