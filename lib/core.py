from lib.peripherals import PeripheralManager
from lib.scheduling import ScheduledFunc, LedFlasher
from lib.utils import connect_wifi, load_env_vars
from lib.networking import setup_mqtt_client, get_default_topic, pack_payload
from lib.decoding import CCA

from ulab import numpy as np
import ujson as json
import uasyncio
from machine import freq
import gc

from micropython import mem_info

ADC_SAMPLE_FREQ = 85 # sample freq in Hz. Only needs to be 2x highest freq in signal (around 40Hz in our case)

env_vars = load_env_vars(".env")

periph_manager = PeripheralManager()

callback = lambda : print("cb")
data_scheduler = ScheduledFunc(0, 2)

mqtt_client = setup_mqtt_client()
topic = get_default_topic()

stim_freqs = [7, 10, 12]
cca_decoder = CCA(stim_freqs, ADC_SAMPLE_FREQ)

def initialise(offline=False):
    
    freq(240000000) # set the CPU frequency to 240 MHz

    # init ADC, SPI and GPIOs
    periph_manager.init()
    
    if not offline:
    
        # connect WiFI
        ssid = env_vars.get("WIFI_SSID")
        password = env_vars.get("WIFI_PASSWORD")
        connect_wifi(ssid, password)
        
        mqtt_client.connect()
        
        msg = "ESP32 client {0} connected".format(mqtt_client.client_id)
        mqtt_client.publish(topic=topic, msg=json.dumps({"message": msg}), qos=1)
    
    # setup garbage collector
    gc.enable()    
    gc.collect()
    gc.threshold(gc.mem_free() // 4 + gc.mem_alloc())
    
    return periph_manager, mqtt_client, data_scheduler

def read_and_decode():
    data = periph_manager.read_adc_buffer()
    gc.collect()
    data = np.array(data).reshape((1, len(data)))
    gc.collect()
    decoded_result = {freq: round(corr[0], 4) for freq, corr in cca_decoder.compute_corr(data).items()}
    gc.collect()
    return decoded_result
        
async def sample_adc():
    global periph_manager
    while True:
        periph_manager.adc_read_to_buff(size=1)
        await uasyncio.sleep(1/ADC_SAMPLE_FREQ)
        
async def decode_and_transmit():
    global periph_manager
    global mqtt_client
    while True:
        await uasyncio.sleep(2)
        periph_manager.flash_led("green", 10, 0.1)
        decoded_data = read_and_decode()
        gc.collect()
        print(decoded_data)
        # payload = pack_payload(raw_data[:50], decoded_data)
        gc.collect()
        # mqtt_client.publish(topic=topic, msg=payload, qos=1)
        
def run():
    event_loop = uasyncio.get_event_loop()

    event_loop.create_task(sample_adc())
    # adc_flasher = LedFlasher(1, ADC_SAMPLE_FREQ, periph_manager.get_led("red"))
    # adc_flasher.start()
    
    event_loop.create_task(decode_and_transmit())
    try:
        event_loop.run_forever()
    except:
        # adc_flasher.stop()
        periph_manager.write_led("red", 1) # signal error