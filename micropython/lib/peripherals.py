import machine
from lib.scheduling import LedFlasher
from lib.utils import update_buffer

import utime as time
import gc

DEFAULT_SPI_PARAMS = {
    "spi_num": 2,
    "sck": 18,
    "mosi": 23,
    "miso": 19,
    "output_amp_gain": 100,  # value between 0-255 controlling gain of output amplifier
}

DEFAULT_ADC_PARAMS = {
    "adc_pin": 33,
    "atten": machine.ADC.ATTN_11DB,
    "width": machine.ADC.WIDTH_12BIT,
    "buffer_size": 256,
}

DEFAULT_LED_CONFIG = {"green": 26, "red": 13}

DEFAULT_BTN_CONFIG = {"btn_a": 32, "btn_b": 34}


DEFAULT_BTN_CONFIG = {"btn_a": 32, "btn_b": 34}

del machine  # only needed for above imports
gc.collect


class PeripheralManager:
    def __init__(
        self,
        adc_params=None,
        spi_params=None,
        led_config=None,
        btn_config=None,
        verbose=True,
    ):
        from machine import Pin

        self.verbose = verbose
        self._adc_params = DEFAULT_ADC_PARAMS
        self._spi_params = DEFAULT_SPI_PARAMS
        self._led_config = DEFAULT_LED_CONFIG
        self._btn_config = DEFAULT_BTN_CONFIG

        self._adc_params.update(adc_params or {})
        self._spi_params.update(spi_params or {})
        self._led_config.update(led_config or {})
        self._btn_config.update(btn_config or {})

        # init LEDs
        self.leds = {}
        for label, pin in self._led_config.items():
            self.leds[label] = Pin(pin, Pin.OUT)

        # init buttons
        self.buttons = {}
        for label, pin in self._btn_config.items():
            self.buttons[label] = Pin(pin, Pin.IN)

        self._adc_buffer = []
        self._timing_buffer = []
        self._adc_scheduler = None

        gc.collect()

    def init(self):
        from machine import Pin, SPI, ADC

        self._adc = ADC(
            Pin(self._adc_params["adc_pin"])
        )  # create ADC object on GPIO 33
        self._adc.atten(self._adc_params["atten"])
        self._adc.width(self._adc_params["width"])
        if self.verbose:
            print("ADC initialised")

        # define spi pins and init
        # spi clock frequency = 10 MHz, clock idle status = LOW
        # spi clock (SCK) pin = GPIO 18
        # spi master output slave input (mosi) pin  = GPIO 23
        # spi master input slave output (miso) pin  = GPIO 19
        # local digiPot select pin = GPIO 5
        # set up button pin = GPIO 21
        get_param = lambda key: Pin(self._spi_params[key])
        temp_spi_params = {key: get_param(key) for key in ["sck", "miso", "mosi"]}
        self._spi = SPI(
            self._spi_params["spi_num"],
            baudrate=10000000,
            polarity=0,
            phase=0,
            **temp_spi_params
        )
        if self.verbose:
            print("SPI initialised")

        output_gain = max(min(self._spi_params["output_amp_gain"], 255), 0)
        self.spi_write(output_gain)
        if self.verbose:
            print(
                "DigiPot set to {0} = gain of {1}".format(
                    output_gain, 1.745 + (255 - output_gain) / (19.2 - 1.745)
                )
            )

        gc.collect()

    @property
    def adc_running(self):
        return (
            not self._adc_scheduler is None
        )  # best proxy we have: no native micropython method

    def flash_led(self, label, freq, duration_sec):
        led = self.get_led(label)
        init_state = led.value()
        flasher = LedFlasher(0, freq, led)
        flasher.run_for_duration(duration_sec)
        led.value(init_state)

    def write_led(self, label, val):
        led = self.get_led(label)
        led.value(val)

    def read_btn(self, label):
        btn = self.get_btn(label)
        return btn.value()

    def get_led(self, label):
        if label not in self.leds:
            raise ValueError(
                "LED with label {0} not found. Valid options are: {1}".format(
                    label, self.leds.keys()
                )
            )
        return self.leds[label]

    def get_btn(self, label):
        if label not in self.buttons:
            raise ValueError(
                "Button with label {0} not found. Valid options are: {1}".format(
                    label, self.buttons.keys()
                )
            )
        return self.buttons[label]

    def get_adc(self):
        return self._adc

    def get_spi(self):
        return self._spi

    def read_adc_buffer(self):
        return self._adc_buffer

    def read_timing_buffer(self):
        return self._timing_buffer

    def spi_write(self, payload):
        # data must be in list format, and can be of arbitrary length e.g. [0x00,0x01,x0x02...etc]
        # devices can be added ad infinitum, with a unique id e.g. "lo" or "mix"
        data = bytearray([17, payload])
        self._spi.write(data)

    def adc_read_to_buff(self, size=-1, record_timing=False):
        """Read adc and write to internal buffer.

        Args:
            size (int, optional): number samples to take. If -1, buffer will be filled. Defaults to -1.
        """
        if record_timing:
            delta = time.ticks_us()

            self._timing_buffer = update_buffer(self._timing_buffer, delta, 20)

        buff_size = self._adc_params["buffer_size"]
        if size < 0 or size > buff_size:
            size = buff_size
        data = [self._adc.read() for i in range(size)]
        self._adc_buffer = update_buffer(self._adc_buffer, data, buff_size)

    def adc_read(self, size=1):
        return [self._adc.read() for i in range(size)]
