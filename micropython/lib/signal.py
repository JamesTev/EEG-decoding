from ulab import numpy as np
from ulab import scipy as spy

# elliptical 10th order bandpass filter with corner frequencies at (4, 28)Hz
# 0.2dB passband ripple and 60dB stopband atten
SOS_SSVEP_BANDPASS_256HZ = np.array(
    [
        [
            5.18206655e-04,
            5.90798154e-04,
            5.18206655e-04,
            1.00000000e00,
            -1.58702496e00,
            6.47839164e-01,
        ],
        [
            1.00000000e00,
            -6.71623649e-01,
            1.00000000e00,
            1.00000000e00,
            -1.56165946e00,
            7.42958422e-01,
        ],
        [
            1.00000000e00,
            -1.19857302e00,
            1.00000000e00,
            1.00000000e00,
            -1.53434838e00,
            8.53019872e-01,
        ],
        [
            1.00000000e00,
            -1.36458617e00,
            1.00000000e00,
            1.00000000e00,
            -1.52074631e00,
            9.31081209e-01,
        ],
        [
            1.00000000e00,
            -1.41818431e00,
            1.00000000e00,
            1.00000000e00,
            -1.52570486e00,
            9.80262684e-01,
        ],
    ]
)

SOS_SSVEP_BANDPASS_128HZ = np.array(
    [
        [0.01296177, 0.02139995, 0.01296177, 1.0, -1.12887027, 0.49738264],
        [1.0, 0.49730727, 1.0, 1.0, -0.70231689, 0.68572112],
        [1.0, -1.99937858, 1.0, 1.0, -1.76256882, 0.80388357],
        [1.0, -0.00419373, 1.0, 1.0, -0.46435738, 0.86883569],
        [1.0, -1.99609143, 1.0, 1.0, -1.93461515, 0.94994846],
        [1.0, -0.15760301, 1.0, 1.0, -0.37566405, 0.96618631],
        [1.0, -1.99348136, 1.0, 1.0, -1.97439955, 0.98509156],
        [1.0, -1.99240012, 1.0, 1.0, -1.98705134, 0.99658552],
    ]
)


def sos_filter(x, sos_coeffs=None, fs=256):
    if sos_coeffs is None:
        if fs == 256:
            sos_coeffs = SOS_SSVEP_BANDPASS_256HZ
        elif fs == 128:
            sos_coeffs = SOS_SSVEP_BANDPASS_128HZ
        else:
            raise ValueError(
                "Unepexcted sampling frequency. Only have SOS filter weights for fs = 64Hz or 256Hz"
            )

    return spy.signal.sosfilt(sos_coeffs, x)
