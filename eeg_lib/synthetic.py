"""
A set of functions to generate synthetic signals that mimic EEG signals. In particular, SSVEP sinsusoidal signals embedded in noise.
"""
from .utils import standardise_ssvep_tensor
import numpy as np


def synth_x(f, Ns, noise_power=0.5, fs=250):
    """
    generate a synthetic signal vector

    args:
    Ns [int]: number of samples (time samples)
    noise_power [float]: variance of WGN noise distribution
    """
    t = np.arange(0, Ns / fs, step=1 / fs)
    return np.sin(2 * np.pi * f * t) + np.random.normal(
        size=Ns, loc=0, scale=noise_power
    )


def synth_X(f, Nc, Ns, noise_power=0.5, fs=200, f_std=0.02, noise_std=0.2):
    """
    Generate a matrix of several variations of the same target signal. This is used
    to simulate the measurement of a common signal over multiple EEG channels
    that have different SNR characteristics.

    args:
    f [float]: target frequency of synthetic signal (Hz)
    Nc [int]: number of channels
    Ns [int]: number of samples (time samples)
    noise_power [float]: variance of WGN noise distribution
    fs [float]: sampling frequency (Hz)
    f_std [float]: standard dev. of freq. in generated signal across channels to simulate interference from other frequency components over different channels
    noise_std [float]: standard dev. of noise across channels
    """
    X = []
    for i in range(Nc):  # simulate noisy sinusoids with varying SNR across Nc channels
        f_i = f * (1 + np.random.normal(scale=f_std))
        sigma_i = noise_power * (1 + np.random.normal(scale=noise_std))
        x = synth_x(f_i, Ns, noise_power=sigma_i)

        x += 0.2 * synth_x(
            f_i * 1.05, Ns
        )  # add extraneous neighbouring signals (task unrelated)
        x += 0.1 * synth_x(f_i * 1.1, Ns)
        X.append(x)

    return np.array(X)


def synth_data_tensor(stim_freqs, Ns, Nc, Nt, noise_power, fs, Nh=3):
    """
    Generate a synthetic 4th order tensor (Chi) of dim. Nf x Nc x Ns x Nt

    args:
    stim_freqs [float]: stimulus frequencies of interest (SSVEP). `Nf = len(stim_freqs)`
    Nc [int]: number of channels
    Ns [int]: number of samples (time samples)
    Nt [int]: number of trials
    Nh [int]: number of harmonics in sinusoidal ref. signal
    noise_power [float]: variance of WGN noise distribution
    fs [float]: sampling frequency (Hz)
    """
    out_tensor = []
    for f in stim_freqs:
        out_tensor.append(
            np.array(
                [synth_X(f, Nc, Ns, noise_power=noise_power) for i in range(Nt)]
            ).transpose(1, 2, 0)
        )
    X = np.array(out_tensor)
    return standardise_ssvep_tensor(X)
