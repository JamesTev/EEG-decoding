import numpy as np

dB = lambda x: 10*np.log10(x) # convert mag. to dB

def save_data_npz(fname, data):
    
    if not isinstance(data, np.ndarray):
        data = data.values
    
    np.savez(fname, data=data)
    
def load_df(fname, key='data', cols=None):
    if cols is None:
        cols=[f'chan{i}' for i in range(1,5)]
    df = pd.DataFrame(np.load(fname)[key], columns=cols)
    return df
