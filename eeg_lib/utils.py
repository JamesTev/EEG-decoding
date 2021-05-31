import numpy as np

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

def resample(X, factor):
    idx_rs = np.arange(0, len(X)-1, factor)
    return X[idx_rs]