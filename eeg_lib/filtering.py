from scipy.signal import filtfilt, iirnotch, freqz, butter, iirfilter
import numpy as np

# Filter requirements.

def butterworth_lowpass(x, fc, fs, order):
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = fc / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, x)
    return y

def iir_bandpass(x, f0, f1, fs, order=4, ftype='cheby2'):
    b, a = iirfilter(order, [f0, f1], rs=60, 
                        btype='band', fs=fs, ftype=ftype, output='ba')
    y = filtfilt(b, a, x)
    return y

def iir_notch(y, f0, fs, Q=30):
    w0 = f0/(fs/2)
    Q = 30
    b, a = iirnotch(w0, Q)
    
    # filter response
    w, h = freqz(b, a)
    filt_freq = w*fs/(2*np.pi)
    y_filt = filtfilt(b, a, y)
    
    return y_filt

def butter_notch(y, f0, fs, order=2):
    w0 = [(f0-15)/(fs/2), (f0+15)/(fs/2)]
    b, a = butter(order, w0, btype='bandstop')
    w, h = freqz(b, a)
    filt_freq = w*fs/(2*np.pi)
    y_filt = filtfilt(b, a, y)
    
    return y_filt
