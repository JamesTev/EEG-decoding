from ulab import numpy as np
import ujson as json
import utime as time
from lib.requests import MicroWebCli as requests

import gc

ADC_SAMPLE_FREQ = 256 # sample freq in Hz
RECORDING_LEN_SEC = 4
OVERLAP = 0.8
DOWNSAMPLED_FREQ = 64 # 64 Hz downsampled  to ensure nyquist condition

PREPROCESSING = True # if true, LP filter and downsample

DEFAULT_LOG_SESSION = 'test-{0}'.format(time.ticks_ms())
MODE = 'log'

periph_manager = None
mqtt_client = None

topic = None 
data_scheduler = None

cca_decoder = None

adc_tim = None

def initialise(offline=False, spi_params=None, adc_params=None):
    from machine import freq
    
    global periph_manager
    global mqtt_client
    global topic
    global cca_decoder
    
    freq(240000000) # set the CPU frequency to 240 MHz

    from lib.peripherals import PeripheralManager
    
    # init ADC, SPI and GPIOs
    periph_manager = PeripheralManager(spi_params=spi_params, adc_params=adc_params)
    periph_manager.init()
    
    from lib.decoding import CCA
    
    cca_ref_freq = ADC_SAMPLE_FREQ
    if PREPROCESSING == True:
        cca_ref_freq = DOWNSAMPLED_FREQ
   
    stim_freqs = [7, 10, 12]
    cca_decoder = CCA(stim_freqs, cca_ref_freq) # !! use DOWNSAMPLED_FREQ for harmonic reference if using downsampling on original signal
    
    if not offline:
        from lib.utils import connect_wifi, load_env_vars
        from lib.networking import setup_mqtt_client, get_default_topic
        
        mqtt_client = setup_mqtt_client()
        topic = get_default_topic()

        env_vars = load_env_vars("lib/.env")
        # connect WiFI
        ssid = env_vars.get("WIFI_SSID")
        password = env_vars.get("WIFI_PASSWORD")
        connect_wifi(ssid, password)
        
        mqtt_client.connect()
        
        msg = "ESP32 client {0} connected".format(mqtt_client.client_id)
        mqtt_client.publish(topic=topic, msg=json.dumps({"message": msg}), qos=1)
    
    # setup scheduler to periodically send data to client
    from lib.scheduling import ScheduledFunc
    data_scheduler = ScheduledFunc(0, 2)
    
    # # setup garbage collector
    gc.enable()    
    gc.collect()
    gc.threshold(gc.mem_free() // 4 + gc.mem_alloc())
    return periph_manager, mqtt_client, data_scheduler

def preprocess_data(signal, downsample_freq=None):
    
    """Preprocess incoming signal before decoding algorithms.
    This involves applying a bandpass filter to isolate the target SSVEP range
    and then downsampling the signal to the Nyquist boundary.
    
    Returns:
        [np.ndarray]: filtered and downsampled signal
    """
    from lib.signal import sos_filter
    downsample_freq = downsample_freq or DOWNSAMPLED_FREQ
    ds_factor = ADC_SAMPLE_FREQ//downsample_freq
    return sos_filter(signal)[::ds_factor]

def read_and_decode(preprocessing=True):
    global cca_decoder
    
    data = periph_manager.read_adc_buffer()
    if len(data) <= 1:
        return {freq: np.nan for freq in stim_freqs}
    gc.collect()
    if preprocessing:
        data = preprocess_data(data)
        gc.collect()
    data = np.array(data).reshape((1, len(data)))
    gc.collect()
    decoded_result = {freq: round(corr[0], 4) for freq, corr in cca_decoder.compute_corr(data).items()}
    gc.collect()
    return decoded_result
        
        
def web_log_callback(*args, **kwargs):
    adc_tim.deinit()
    periph_manager.write_led("green", 1)
    data = periph_manager.read_adc_buffer()
    data = preprocess_data(data, downsample_freq = DOWNSAMPLED_FREQ)
    if data is not None:
        packed_data = {'data': list(data), 'timestamp': time.ticks_us(), 'session_id': DEFAULT_LOG_SESSION}
        requests.POSTRequest('http://james-tev.local:5000/', packed_data)

    periph_manager.write_led("green", 0)
    adc_tim.init(freq=ADC_SAMPLE_FREQ, callback=sample_callback)
    
def sample_callback(*args, **kwargs):
    periph_manager.adc_read_to_buff(size=1)
        
def run():
    from machine import Pin, Timer
    from lib.networking import pack_payload
    
    global periph_manager
    global mqtt_client
    global topic

    # start sampling on timer 0
    adc_tim = Timer(0)
    adc_tim.init(freq=ADC_SAMPLE_FREQ, callback=sample_callback)

    # flash LED
    periph_manager.write_led("green", 1)

    time.sleep(RECORDING_LEN_SEC) # wait for at least one full window
    
    if MODE != 'log':
        while True:
            time.sleep(RECORDING_LEN_SEC*(1-OVERLAP))
            decoded_data = read_and_decode(preprocessing=PREPROCESSING)
            gc.collect()
            print(decoded_data)
            payload = pack_payload([1,2,3], decoded_data, client_id=mqtt_client.client_id)
            gc.collect()
            mqtt_client.publish(topic=topic, msg=payload, qos=1)
    else:
        # init web logging 
        periph_manager.write_led("green", 0)
        buff_size = 256
        log_freq = 0.5*(DOWNSAMPLED_FREQ/buff_size)
        log_tim = Timer(1)
        log_tim.init(freq=log_freq, callback=web_log_callback)
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            log_tim.deinit()
            adc_tim.deinit()
            print("received keyboard interrupt")