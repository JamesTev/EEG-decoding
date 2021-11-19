from .peripherals import PeripheralManager
from ulab import numpy as np

import gc
import config

# enable and configure garbage collection
gc.enable()
gc.collect()
gc.threshold(gc.mem_free() // 4 + gc.mem_alloc())


class BaseRunner:
    def __init__(self, stimulus_freqs=None) -> None:
        if stimulus_freqs is None:
            self.stim_freqs = config.STIM_FREQS  # assign defaults

        self.base_sample_freq = config.ADC_SAMPLE_FREQ
        self.downsampled_freq = config.DOWNSAMPLED_FREQ

        self.preprocessing_enabled = config.PREPROCESSING

        if self.preprocessing_enabled:
            self.downsampled_freq = config.DOWNSAMPLED_FREQ

        self.sample_counter = 0
        self.buffer_size = 256  # TODO: update this to be populated dynamically

        self.output_buffer = [0.0 for i in range(self.buffer_size)]

    def setup(self, spi_params=None, adc_params=None):
        from machine import freq

        freq(config.BASE_CLK_FREQ)  # set the CPU frequency

        self._init_decoder()
        gc.collect()

        self._init_peripherals(spi_params, adc_params)
        gc.collect()

    def preprocess_data(self, signal):

        """Preprocess incoming signal before decoding algorithms.
        This involves applying a bandpass filter to isolate the target SSVEP range
        and then downsampling the signal to the Nyquist boundary.

        Returns:
            [np.ndarray]: filtered and downsampled signal
        """
        from lib.signal import sos_filter

        ds_factor = self.downsampling_factor
        signal = np.array(signal) - np.mean(signal)  # remove DC component

        # downsample filtered signal by only selecting every `ds_factor` sample
        return sos_filter(signal, fs=self.base_sample_freq)[::ds_factor]

    def read_and_decode(self):

        data = self._read_internal_buffer(preprocess=self.preprocessing_enabled)
        if len(data) <= 1:
            return {freq: np.nan for freq in self.stim_freqs}
        gc.collect()

        data = np.array(data).reshape((1, len(data)))
        gc.collect()

        result = self.decoder.compute_corr(data)

        decoded_result = {freq: round(corr[0], 5) for freq, corr in result.items()}
        gc.collect()
        return decoded_result

    def decode(self):
        data = np.array(self.output_buffer)
        data = data.reshape((1, len(data)))  # reshape to row vector
        gc.collect()

        result = self.decoder.compute_corr(data)

        decoded_result = {freq: round(corr[0], 5) for freq, corr in result.items()}
        gc.collect()
        return decoded_result

    def sample_callback(self, *args, **kwargs):
        from lib.utils import update_buffer

        self.periph_manager.adc_read_to_buff(size=1)
        self.sample_counter += 1

        # this will only be true every 1s once buffer fills
        if self.sample_counter >= self.buffer_size:
            self.periph_manager.write_led("red", 1)
            data = self._read_internal_buffer(preprocess=self.preprocessing_enabled)
            self.output_buffer = update_buffer(
                self.output_buffer, list(data), self.buffer_size
            )
            self.sample_counter = 0
            self.periph_manager.write_led("red", 0)

    def read_output_buffer(self):
        return self.output_buffer

    def start_sample_timer(self):
        from machine import Timer

        self.sample_timer = Timer(0)
        self.sample_timer.init(
            freq=self.base_sample_freq, callback=self.sample_callback
        )

    def stop_sample_timer(self):
        if self.sample_timer is not None:
            self.sample_timer.deinit()

    @property
    def downsampling_factor(self):
        return self.base_sample_freq // self.downsampled_freq

    def _read_internal_buffer(self, preprocess=False):
        data = self.periph_manager.read_adc_buffer()
        if preprocess and len(data) > 1:
            data = self.preprocess_data(data)
        return data

    def _init_peripherals(self, spi_params, adc_params):

        self.periph_manager = PeripheralManager(
            spi_params=spi_params, adc_params=adc_params
        )
        self.periph_manager.init()
        self.start_sample_timer()

    def _init_decoder(self):
        from lib.decoding import CCA

        # note: downsampled_freq is same as base sampling freq if
        # preprocessing is disabled
        self.decoder = CCA(self.stim_freqs, self.downsampled_freq)


class Runner(BaseRunner):
    def setup(self, spi_params=None, adc_params=None, log_period=5):
        super().setup(spi_params=spi_params, adc_params=adc_params)
        gc.collect()

        self.configure_wifi(env_path="lib/.env")
        gc.collect()

        self.log_period = log_period
        self.start_logger()

    def web_log_callback(self, *args, **kwargs):
        from lib.requests import MicroWebCli as requests
        import utime as time

        # pause timer while async request completes
        self.log_timer.deinit()

        self.periph_manager.write_led("green", 1)
        packed_data = {
            "data": self.output_buffer,
            "timestamp": time.ticks_us(),
            "session_id": self.log_session,
        }
        requests.POSTRequest(self.log_url, packed_data)

        self.periph_manager.write_led("green", 0)

        # restart log timer
        self.log_timer.init(freq=self.log_freq, callback=self.web_log_callback)

    def start_logger(self):
        from machine import Timer

        self.log_timer = Timer(1)
        self.log_session = config.DEFAULT_LOG_SESSION
        self.log_url = config.LOG_URL

        self.log_timer.init(freq=self.log_freq, callback=self.web_log_callback)

    def stop_logger(self):
        if self.log_timer is not None:
            self.log_timer.deinit()

    @staticmethod
    def configure_wifi(env_path=".env"):

        from lib.utils import connect_wifi, load_env_vars

        env_vars = load_env_vars(env_path)
        ssid = env_vars.get("WIFI_SSID")
        password = env_vars.get("WIFI_PASSWORD")
        connect_wifi(ssid, password)
