class Enum(set):
    """
    Class to simulate the behaviour of an enum type
    """

    def __getattr__(self, name):
        if name in self:
            return name
        raise AttributeError


def delay_ms(t):
    import utime

    t0 = utime.time()
    while (utime.time() - t0) * 1000 < t:
        pass  # TODO: investigate if this will actually free core during delay


def update_buffer(buf, el, max_size, inplace=True):
    if max_size != len(buf):
        inplace = False

    if type(el) in [float, int]:
        el = [el]
    el = el[-max_size:]
    tmp = (buf[-(max_size - len(el)) :] + el)[-max_size:]
    if inplace:
        for i, el in enumerate(tmp):
            buf[i] = el
        return None
    return tmp


def connect_wifi(ssid, password):
    import network

    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    if not wlan.isconnected():
        print("connecting to network...")
        wlan.connect(ssid, password)
        while not wlan.isconnected():  # okay that this is blocking
            pass
    print("network config:", wlan.ifconfig())
    return wlan


def load_env_vars(path):
    import ure as re

    envre = re.compile(r"""^([^\s=]+)=(?:[\s"']*)(.+?)(?:[\s"']*)$""")
    result = {}
    with open(path) as ins:
        for line in ins:
            match = envre.match(line)
            if match is not None:
                result[match.group(1)] = match.group(2).replace("\n", "")
    return result


def write_json(filename, data):
    import ujson as json

    with open(filename, "w") as f:
        json.dump(data, f)


def read_json(filename):
    import ujson as json

    with open(filename) as f:
        return json.load(f)
