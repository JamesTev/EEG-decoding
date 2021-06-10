from scipy.signal import filtfilt, iirnotch, freqz, butter, iirfilter
import numpy as np
import warnings
import scipy.signal


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

def iir_bandpass_ssvep_tensor(X, f0, f1, fs, order=4, ftype='cheby2'):
    """
    Perform IIR bandpass over SSVEP frequency band.
    
    Expects 4th order data tensor X in form: Nf x Nc X Ns x Nt
    """
    X_f = np.zeros_like(X)
    Nf, Nc, Ns, Nt = X.shape
    
    bp_filt = lambda x: iir_bandpass(x, f0, f1, fs, order=order, ftype=ftype)

    for trial in range(Nt): # TODO parallelise or similar
        for f_idx in range(Nf):
            X_i = X[f_idx, :, :, trial]
            X_f[f_idx, :, :, trial] = np.array([bp_filt(X_i[chan, :]) for chan in range(Nc)])
    return X_f

def filterbank(eeg, fs, idx_fb):    
    if idx_fb == None:
        warnings.warn('stats:filterbank:MissingInput '\
                      +'Missing filter index. Default value (idx_fb = 0) will be used.')
        idx_fb = 0
    elif (idx_fb < 0 or 9 < idx_fb):
        raise ValueError('stats:filterbank:InvalidInput '\
                          +'The number of sub-bands must be 0 <= idx_fb <= 9.')
            
    if (len(eeg.shape)==2):
        num_chans = eeg.shape[0]
        num_trials = 1
    else:
        num_chans, _, num_trials = eeg.shape
    
    # Nyquist Frequency = Fs/2N
    Nq = fs/2
    
    passband = [6, 14, 22, 30, 38, 46, 54, 62, 70, 78]
    stopband = [4, 10, 16, 24, 32, 40, 48, 56, 64, 72]
    Wp = [passband[idx_fb]/Nq, 90/Nq]
    Ws = [stopband[idx_fb]/Nq, 100/Nq]
    [N, Wn] = scipy.signal.cheb1ord(Wp, Ws, 3, 40) # band pass filter StopBand=[Ws(1)~Ws(2)] PassBand=[Wp(1)~Wp(2)]
    [B, A] = scipy.signal.cheby1(N, 0.5, Wn, 'bandpass') # Wn passband edge frequency
    
    y = np.zeros(eeg.shape)
    if (num_trials == 1):
        for ch_i in range(num_chans):
            #apply filter, zero phass filtering by applying a linear filter twice, once forward and once backwards.
            # to match matlab result we need to change padding length
            y[ch_i, :] = scipy.signal.filtfilt(B, A, eeg[ch_i, :], padtype = 'odd', padlen=3*(max(len(B),len(A))-1))
        
    else:
        for trial_i in range(num_trials):
            for ch_i in range(num_chans):
                y[ch_i, :, trial_i] = scipy.signal.filtfilt(B, A, eeg[ch_i, :, trial_i], padtype = 'odd', padlen=3*(max(len(B),len(A))-1))     
    return y