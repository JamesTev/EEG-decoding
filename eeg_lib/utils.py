import numpy as np
import glob

dB = lambda x: 10*np.log10(x) # convert mag. to dB

def save_data_npz(fname, data, **kwargs):
    
    if not isinstance(data, np.ndarray):
        data = data.values
    
    np.savez(fname, data=data, **kwargs)
    
def load_df(fname, key='data', cols=None):
    if cols is None:
        cols=[f'chan{i}' for i in range(1,5)]
    df = pd.DataFrame(np.load(fname)[key], columns=cols)
    return df

def standardise(X):
    axis = np.argmax(X.shape)
    return (X-np.mean(X, axis=axis))/np.std(X, axis=axis)

def standardise_ssvep_tensor(X):
    # Given a obs matrix for given f, and trial, rows (channels) should all be zero mean and unit std dev

    Nf, Nc, Ns, Nt = X.shape

    for n in range(Nf):
        for t in range(Nt):
            obs = X[n, :, :, t]
            mu = np.broadcast_to(obs.mean(axis=1), (Ns, Nc)).T
            sigma = np.broadcast_to(obs.std(axis=1), (Ns, Nc)).T
            X[n, :, :, t] = (obs-mu)/sigma
    
    return X


def resample(X, factor):
    idx_rs = np.arange(0, len(X)-1, factor)
    return X[idx_rs]

def load_trials(path_pattern, verbose=False):
    all_files = glob.glob(path_pattern)
    data = []
    
    min_len = 10e6 # trials lengths will be very similar but may differ by 1 or 2 samples
    for filename in all_files:
        if verbose:
            print(f"Loading file {filename}")
        f = np.load(filename, allow_pickle=True)
        data.append(f['data'])
        if len(f['data']) < min_len:
            min_len = len(f['data'])
    
    return np.array([trial[:min_len-1] for trial in data])