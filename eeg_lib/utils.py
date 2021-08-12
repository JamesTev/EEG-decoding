import numpy as np
import glob
import json

from numpy import linalg as LA

dB = lambda x: 10*np.log10(x) # convert mag. to dB

def solve_gen_eig_prob(A, B, eps=1e-8):
    """
    Solves the generalised eigenvalue problem of the form:
    Aw = \lambda*Bw
    
    Note: can be validated against `scipy.linalg.eig(A, b=B)`
    
    Ref: 
    'Eigenvalue and Generalized Eigenvalue Problems: Tutorial (2019)'
    Benyamin Ghojogh and Fakhri Karray and Mark Crowley
    arXiv 1903.11240

    """
    Lam_b, Phi_b = LA.eig(B) # eig decomp of B alone
    Lam_b = np.eye(len(Lam_b))*Lam_b # convert to diagonal matrix of eig vals
    
    Lam_b_sq = np.nan_to_num(Lam_b**0.5)+eps*np.eye(len(Lam_b))
    Phi_b_hat = Phi_b.dot(LA.inv(Lam_b_sq))
    A_hat = Phi_b_hat.T.dot(A).dot(Phi_b_hat)
    Lam_a, Phi_a = LA.eig(A_hat)
    Lam_a = np.eye(len(Lam_a))*Lam_a
    
    Lam = Lam_a
    Phi = Phi_b_hat.dot(Phi_a)
    
    return np.diag(Lam), Phi
    
def inv_square(A):
    """
    Compute inverse square root of a matrix using Cholesky decomp.
    
    Requires A to be positive definite.
    """
    return LA.inv(LA.cholesky(A))

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

def write_json(filename, data):
    with open(filename, 'w') as f:
            json.dump(data, f)

def read_json(filename):
    with open(filename) as f:
        return json.load(f)