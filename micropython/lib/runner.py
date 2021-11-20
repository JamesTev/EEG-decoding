from .peripherals import PeripheralManager
from ulab import numpy as np
from logging import BaseLogger, MQTTLogger, HTTPLogger, logger_types

import gc
import config

from micropython import schedule
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
        self.decoded_output = {}

        self.is_setup = False
        self.is_sampling = False
        self.is_logging = False

    def setup(self, spi_params=None, adc_params=None, log_period=5, logger_type=None):
        from machine import freq

        freq(config.BASE_CLK_FREQ)  # set the CPU frequency

        self._init_decoder()
        gc.collect()

        self._init_peripherals(spi_params, adc_params)
        gc.collect()

        self._setup_logger(log_period, logger_type)
        gc.collect()

        self.is_setup = True

    def run(self):
        if not self.is_setup:
            raise ValueError("Runner not setup. Call `.setup()` before running.")
            
        self.start_sample_timer()

        if self.logger is not None:
            self.start_logger()

    def stop(self):
        if self.is_sampling:
            self.stop_sample_timer()

        if self.is_logging:
            self.stop_logger()

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

    def decode(self, *args):
        """
        Run decoding on current state of output buffer.

        Note that `*args` is specified but not used directly: this allows
        this function to be called using `micropython.schedule` which
        requires the scheduled func to accept an argument.
        """
        data = np.array(self.output_buffer)
        data = data.reshape((1, len(data)))  # reshape to row vector
        gc.collect()

        result = self.decoder.compute_corr(data)

        # note: need to be careful not to change the memory address of this variable using direct 
        # assignment since the logger depends on this reference. Also would just be inefficient.
        self.decoded_output.update({freq: round(corr[0], 5) for freq, corr in result.items()})
        gc.collect()
        return self.decoded_output

    def sample_callback(self, *args, **kwargs):
        from lib.utils import update_buffer

        self.periph_manager.adc_read_to_buff(size=1)
        self.sample_counter += 1

        # this will only be true every 1s once buffer fills
        if self.sample_counter >= self.buffer_size:
            self.periph_manager.write_led("red", 1)
            data = self._read_internal_buffer(preprocess=self.preprocessing_enabled)
            update_buffer(
                self.output_buffer, list(data), self.buffer_size, inplace=True
            )
            self.sample_counter = 0
            self.periph_manager.write_led("red", 0)

            # TODO: workout how to run decoding in another handler as 
            # this could take a non-negligible amount of time which
            # would disrupt consistency of sampling freq. For now,
            # we can schedule this function to run 'soon' while allowing other
            # ISRs to interrupt it if need be.
            try:
                schedule(self.decode, None)
            except RuntimeError:
                # if schedule queue is full, run now
                self.decode()

    def read_output_buffer(self):
        return self.output_buffer

    def start_logger(self):
        if self.logger is not None:
            self.logger.start()
            self.is_logging = True

    def stop_logger(self):
        if self.logger is not None:
            self.logger.stop()
            self.is_logging = False

    def start_sample_timer(self):
        from machine import Timer

        self.sample_timer = Timer(0)
        self.sample_timer.init(
            freq=self.base_sample_freq, callback=self.sample_callback
        )
        self.is_sampling = True

    def stop_sample_timer(self):
        if self.sample_timer is not None:
            self.sample_timer.deinit()
            self.is_sampling = False

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

    def _init_decoder(self):
        from lib.decoding import CCA

        # note: downsampled_freq is same as base sampling freq if
        # preprocessing is disabled
        self.decoder = CCA(self.stim_freqs, self.downsampled_freq)

    def _setup_logger(self, log_period, logger_type):
        if logger_type is not None:
            if logger_type != logger_types.SERIAL:
                print(
                    "Warning: only the `SERIAL` logger type is available offline. Defaulting to this."
                )
            self.logger = BaseLogger(
                log_period, self.decoded_output, self.output_buffer
            )
        else:
            self.logger = None


class OnlineRunner(BaseRunner):
    def setup(
        self,
        spi_params=None,
        adc_params=None,
        log_period=5,
        logger_type=None,
        **logger_params
    ):
        super().setup(spi_params=spi_params, adc_params=adc_params)
        gc.collect()

        self.configure_wifi(env_path="lib/.env")
        gc.collect()

        self._setup_logger(log_period, logger_type, **logger_params)

    def _setup_logger(self, log_period, logger_type, **logger_params):
        if logger_type is not None:
            base_logger_args = [log_period, self.decoded_output, self.output_buffer]
            if logger_type == logger_types.MQTT:
                self.logger = MQTTLogger(*base_logger_args, **logger_params)
            elif logger_type == logger_types.HTTP:
                self.logger = HTTPLogger(*base_logger_args, **logger_params)
            else:
                self.logger = BaseLogger(*base_logger_args)
        else:
            self.logger = None

    @staticmethod
    def configure_wifi(env_path=".env"):

        from lib.utils import connect_wifi, load_env_vars

        env_vars = load_env_vars(env_path)
        ssid = env_vars.get("WIFI_SSID")
        password = env_vars.get("WIFI_PASSWORD")
        connect_wifi(ssid, password)
