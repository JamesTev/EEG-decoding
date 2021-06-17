
from ulab import numpy as np

from .computation import corr, max_eig, standardise
    
class CCA():
    
    def __init__(self, stim_freqs, fs, Nh=2):
        self.Nh = Nh
        self.stim_freqs = stim_freqs
        self.fs = fs
        
    def compute_corr(self, X_test):            
        result = {}
        Cxx = np.dot(X_test, X_test.transpose()) # precompute data auto correlation matrix
        for f in self.stim_freqs:
            Y = harmonic_reference(f, self.fs, np.max(X_test.shape()), Nh=self.Nh, standardise_out=True)
            rho = self.cca_eig(X_test, Y, Cxx=Cxx) # canonical variable matrices. Xc = X^T.W_x
            result[f] = rho
        return result
    
    @staticmethod
    def cca_eig(X, Y, Cxx=None):
        if Cxx is None:
            Cxx = np.dot(X, X.transpose()) # auto correlation matrix
        Cyy = np.dot(Y, Y.transpose()) 
        Cxy = np.dot(X, Y.transpose()) # cross correlation matrix
        Cyx = np.dot(Y, X.transpose()) # same as Cxy.T

        M1 = np.dot(np.linalg.inv(Cxx), Cxy) # intermediate result
        M2 = np.dot(np.linalg.inv(Cyy), Cyx)

        lam, _ = max_eig(np.dot(M1, M2))
        return np.sqrt(lam)
    
def harmonic_reference(f0, fs, Ns, Nh=2, standardise_out=False):
    
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

    tidx = np.arange(1,Ns+1)*(1/fs) #time index
    
    tmp = []
    for harm_i in range(1,Nh+1):
        # Sin and Cos
        tmp.extend([np.sin(tidx*2*np.pi*harm_i*f0),
                    np.cos(tidx*2*np.pi*harm_i*f0)])
    y_ref = np.array(tmp)
    if standardise_out: # zero mean, unit std. dev
        return standardise(y_ref)
    return y_ref
