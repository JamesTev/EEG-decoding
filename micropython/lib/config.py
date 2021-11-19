import utime

BASE_CLK_FREQ = 240000000  # 240 MHz for ESP32

ADC_SAMPLE_FREQ = 256  # sample freq in Hz

RECORDING_LEN_SEC = 4

OVERLAP = 0.8

DOWNSAMPLED_FREQ = 64  # 64 Hz downsampled  to ensure nyquist condition

PREPROCESSING = True  # if true, LP filter and downsample

STIM_FREQS = [7, 10, 12]  # stimulus freqs. in Hz

DEFAULT_LOG_SESSION = "test-{0}".format(utime.ticks_ms())

MODE = "log"

HTTP_LOG_URL = "http://james-tev.local:5000/"
