# compute STFT using Welch's method
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fftshift

import pandas as pd
from .utils import dB


def plot_stft_spectra(
    Sxx,
    f,
    ssvep_f0=None,
    figsize=(14, 12),
    recursive_av=True,
    f_ssvep=[8.75, 10, 12, 15],
):
    def plot_spectrum(ax, f, Sxx, is_db=False, ssvep_f0=None, title=""):
        if not is_db:
            Sxx = dB(np.abs(Sxx) ** 2)
        ax.plot(f, Sxx)
        if ssvep_f0 is not None:
            ax.axvline(ssvep_f0, c="r", ls=":", lw=1.5, label="expected SSVEP $f_0$")
        ax.legend()
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (dB)")
        ax.set_title(title)
        return ax

    n_win = Sxx.shape[1]
    fig, axes = plt.subplots(
        n_win,
        3,
        figsize=figsize,
    )
    for i in range(n_win):
        Sxx_av = np.mean(Sxx[:, 0 : i + 1], axis=1)
        ax_psd = plot_spectrum(
            axes[i][0], f, Sxx[:, i], ssvep_f0=ssvep_f0, title=f"Window {i+1}"
        )
        ax_zoom = plot_spectrum(
            axes[i][1], f, Sxx[:, i], ssvep_f0=ssvep_f0, title=f"Window {i+1}"
        )

        if recursive_av:
            plot_spectrum(ax_psd, f, Sxx_av)
            plot_spectrum(ax_zoom, f, Sxx_av)

        ax_zoom.set_xlim(
            min(f_ssvep) * 0.75, max(f_ssvep) * 1.25
        )  # zoom in on freq band of interest
        ax_psd.set_xlim(0, 70)
        width = 0.2
        # first, decode using indepdendent spectra
        f_decoded, p = decode_ssvep(Sxx[:, i], f, f_ssvep)
        axes[i][2].bar(x=f_ssvep - width / 2, height=p, width=width)

        # now, decode using recursively-averaged spectra
        f_decoded, p = decode_ssvep(Sxx_av, f, f_ssvep)
        #         labels=[f'$f_{i}$' for i in range(len(p))]
        axes[i][2].bar(x=f_ssvep + width / 2, height=p, tick_label=f_ssvep, width=width)

        axes[i][2].legend(["P($f_i$) no av.", "P($f_i$) with av."])

    fig.tight_layout(pad=1.5)


def decode_ssvep(Sxx, f, target_freqs, convert_to_mag=True, verbose=False):
    """
    Takes PSD estimate Sxx and freq vector f and finds candidate freq in target_freqs
    that corresponds to max power in Sxx. Also returns list of Sxx values at target_freqs.
    """
    if convert_to_mag:
        Sxx = np.abs(Sxx) ** 2
    p = []  # power vector - store PSD at each freq
    for fsv in target_freqs:
        f_idx = np.searchsorted(f, fsv, side="left")
        if f_idx != fsv and verbose:
            print(
                f"Warning: couldn't find exact match for SSVEP freq {fsv}. Using {f[f_idx]}"
            )
        p.append(Sxx[f_idx])
    f_decoded = target_freqs[np.argmax(p)]
    return f_decoded, p


from scipy.fft import rfft, rfftfreq  # real fft


def plot_periodogram(x, fs, ssvep_f0=None, N=2048, figsize=(14, 10), axes=None):

    N = min(N, len(x) - 1)

    if not isinstance(x, np.ndarray):
        x = x.values
    X = rfft(x, n=N)
    Pxx_fft = np.abs(X) ** 2
    w1 = np.linspace(0, 1, N // 2 + 1)  # norm freq (pi rad/sample)
    f1 = w1 * fs / 2  # pi rad/sample corresponds to fs/2

    welch_wins = [N // 2]
    Pxx_welch_mat = np.zeros((len(welch_wins), N // 2 + 1))

    for i, win in enumerate(welch_wins):
        f_welch, Pxx_welch = signal.welch(
            x, fs, nperseg=win, nfft=N
        )  # nperseg = welch window len
        Pxx_welch_mat[i, :] = dB(Pxx_welch)

    if axes is None:
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=figsize)
    else:
        ax0, ax1 = axes

    ax0.plot(f1, dB(Pxx_fft))
    ax0.set_title("Estimated PSD: Standard Periodogram")

    ax1.plot(f_welch, Pxx_welch_mat.T)
    ax1.set_title("Estiamted PSD: Welch Averaged Periodogram")

    for ax in (ax0, ax1):
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("PSD (dB)")

        x_max = 60
        ax.set_xlim(0, x_max)
        ax.set_xticks(np.arange(0, x_max, step=2))
        if ssvep_f0 is not None:
            ax.axvline(
                ssvep_f0, ls=":", lw=1.5, color="r", label="expected $f^{(0)}_{SSVEP}$"
            )
            print(f"Fundamental SSVEP frequency expected at {ssvep_f0}Hz")
            ax.legend()
        ax.grid()

    plt.tight_layout(pad=1)


def stft(x, Nwin, Nfft=None):
    import numpy.fft as fft

    """
    Short-time Fourier transform: convert a 1D vector to a 2D array
    The short-time Fourier transform (STFT) breaks a long vector into disjoint
    chunks (no overlap) and runs an FFT (Fast Fourier Transform) on each chunk.
    The resulting 2D array can 
    Parameters
    ----------
    x : array_like
        Input signal (expected to be real)
    Nwin : int
        Length of each window (chunk of the signal). Should be ≪ `len(x)`.
    Nfft : int, optional
        Zero-pad each chunk to this length before FFT. Should be ≥ `Nwin`,
        (usually with small prime factors, for fastest FFT). Default: `Nwin`.
    Returns
    -------
    out : complex ndarray
        `len(x) // Nwin` by `Nfft` complex array representing the STFT of `x`.
    
    See also
    --------
    istft : inverse function (convert a STFT array back to a data vector)
    stftbins : time and frequency bins corresponding to `out`
    """
    Nfft = Nfft or Nwin
    Nwindows = x.size // Nwin
    # reshape into array `Nwin` wide, and as tall as possible. This is
    # optimized for C-order (row-major) layouts.
    arr = np.reshape(x[: Nwindows * Nwin], (-1, Nwin))
    stft = fft.rfft(arr, Nfft)
    return stft


def stftbins(x, Nwin, Nfft=None, d=1.0):
    import numpy.fft as fft

    """
    Time and frequency bins corresponding to short-time Fourier transform.
    Call this with the same arguments as `stft`, plus one extra argument: `d`
    sample spacing, to get the time and frequency axes that the output of
    `stft` correspond to.
    Parameters
    ----------
    x : array_like
        same as `stft`
    Nwin : int
        same as `stft`
    Nfft : int, optional
        same as `stft`
    d : float, optional
        Sample spacing of `x` (or 1 / sample frequency), units of seconds.
        Default: 1.0.
    Returns
    -------
    t : ndarray
        Array of length `len(x) // Nwin`, in units of seconds, corresponding to
        the first dimension (height) of the output of `stft`.
    f : ndarray
        Array of length `Nfft`, in units of Hertz, corresponding to the second
        dimension (width) of the output of `stft`.
    """
    Nfft = Nfft or Nwin
    Nwindows = x.size // Nwin
    t = np.arange(Nwindows) * (Nwin * d)
    f = fft.rfftfreq(Nfft, d)
    return t, f
