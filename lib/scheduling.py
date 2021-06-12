from machine import Pin, Timer
import micropython
import ujson as json
from lib.signal import dummy_calc

micropython.alloc_emergency_exception_buf(100)

class ScheduledFunc():
    def __init__(self, timer_num, freq):
        self.freq = freq
        self.tim = Timer(timer_num)
        
    def start(self):
        period = int(1000/self.freq)
        self.tim.init(period=period, callback=self.cb) # 100 ms period
        
    def run_for_duration(self, t):
        import utime
        self.start()
        utime.sleep(t)
        self.stop()
    
    def stop(self):
        self.tim.deinit()
        
    def cb(self, timer, *args):
        pass
        
class LedFlasher(ScheduledFunc):
    def __init__(self, timer_num, freq, led):
        super().__init__(timer_num, freq*2) # so that LED comes on at f=freq
        self.led = led
        
    def cb(self, timer, *args):
        self.led.value(not self.led.value())
        
class WsDataScheduler(ScheduledFunc):
    
    def __init__(self, freq, ws_server, led_pin=5, timer_num=0):
        super().__init__(timer_num, freq)
        self.led = Pin(led_pin, Pin.OUT)
        self.ws_server = ws_server
        
    def start(self):
        super().start()
        self.ws_server.start()
    
    def stop(self):
        super().stop()
        self.ws_server.stop()
        
    def send_data(self, *args):
        self.ws_server.process_all()
        data = dummy_calc() #.flatten()
        self.ws_server.broadcast(json.dumps(data))
#         self.ws_server.broadcast(data.tobytes())
    
    def cb(self, timer, *args):
        self.led.value(not self.led.value())
        micropython.schedule(self.send_data, self) # see https://docs.micropython.org/en/latest/library/micropython.html?highlight=schedule#micropython.schedule