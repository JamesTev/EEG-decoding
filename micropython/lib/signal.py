from ulab import numpy as np
from ulab import scipy as spy

"""
The digital filter coefficients below were designed using Scipy to the same
specification:

Elliptical 10th order bandpass filter with corner frequencies at (4, 28)Hz,
0.2dB passband ripple and 80dB stopband atten
"""

SOS_SSVEP_BANDPASS_256HZ = np.array(
    [
        [
            5.18442631e-04,
            5.91022291e-04,
            5.18442631e-04,
            1.00000000e00,
            -1.58700686e00,
            6.47826110e-01,
        ],
        [
            1.00000000e00,
            -6.71721317e-01,
            1.00000000e00,
            1.00000000e00,
            -1.56164716e00,
            7.42956116e-01,
        ],
        [
            1.00000000e00,
            -1.19862825e00,
            1.00000000e00,
            1.00000000e00,
            -1.53434369e00,
            8.53024717e-01,
        ],
        [
            1.00000000e00,
            -1.36462221e00,
            1.00000000e00,
            1.00000000e00,
            -1.52074686e00,
            9.31086238e-01,
        ],
        [
            1.00000000e00,
            -1.41821305e00,
            1.00000000e00,
            1.00000000e00,
            -1.52570664e00,
            9.80264626e-01,
        ],
    ]
)

SOS_SSVEP_BANDPASS_128HZ = np.array(
    [
        [0.00489814, 0.00882672, 0.00489814, 1.0, -1.12754282, 0.37507747],
        [1.0, 0.89364345, 1.0, 1.0, -0.86464138, 0.5663009],
        [1.0, 0.27438961, 1.0, 1.0, -0.59631233, 0.76500326],
        [1.0, -0.00656791, 1.0, 1.0, -0.4363727, 0.89332053],
        [1.0, -0.11037337, 1.0, 1.0, -0.37229848, 0.96976145],
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
