
from ulab import numpy as np
import gc

from .computation import solve_eig_qr, standardise
    
class CCA():
    
    def __init__(self, stim_freqs, fs, Nh=2):
        self.Nh = Nh
        self.stim_freqs = stim_freqs
        self.fs = fs
        
    def compute_corr(self, X_test):            
        result = {}
        Cxx = np.dot(X_test, X_test.transpose()) # precompute data auto correlation matrix
        for f in self.stim_freqs:
            Y = harmonic_reference(f, self.fs, np.max(X_test.shape), Nh=self.Nh, standardise_out=False)
            rho = self.cca_eig(X_test, Y, Cxx=Cxx) # canonical variable matrices. Xc = X^T.W_x
            result[f] = rho
        return result
    
    @staticmethod
    def cca_eig(X, Y, Cxx=None, eps=1e-6):
        if Cxx is None:
            Cxx = np.dot(X, X.transpose()) # auto correlation matrix
        Cyy = np.dot(Y, Y.transpose()) 
        Cxy = np.dot(X, Y.transpose()) # cross correlation matrix
        Cyx = np.dot(Y, X.transpose()) # same as Cxy.T

        M1 = np.dot(np.linalg.inv(Cxx+eps), Cxy) # intermediate result
        M2 = np.dot(np.linalg.inv(Cyy+eps), Cyx)

        lam, _ = solve_eig_qr(np.dot(M1, M2), 20)
        return np.sqrt(lam)
    
def harmonic_reference(f0, fs, Ns, Nh=1, standardise_out=False):
    
    '''
    Generate reference signals for canonical correlation analysis (CCA)
    -based steady-state visual evoked potentials (SSVEPs) detection [1, 2].
    function [ y_ref ] = cca_reference(listFreq, fs,  Ns, Nh) 
    Input:
      f0        : stimulus frequency
      fs              : Sampling frequency
      Ns              : # of samples in trial
      Nh          : # of harmonics
    Output:
      y_ref           : Generated reference signals with shape (Nf, Ns, 2*Nh)
    '''  
    X = np.zeros((Nh*2, Ns))
    
    for harm_i in range(Nh):
        # Sin and Cos
        X[2*harm_i, :] = np.sin(np.arange(1,Ns+1)*(1/fs)*2*np.pi*(harm_i+1)*f0)
        gc.collect()
        X[2*harm_i+1, :] = np.cos(np.arange(1,Ns+1)*(1/fs)*2*np.pi*(harm_i+1)*f0)
        gc.collect()

    # print(micropython.mem_info(1))
    if standardise_out: # zero mean, unit std. dev
        return standardise(X)
    return X
