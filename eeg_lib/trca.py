import numpy as np
from scipy.stats import pearsonr

# Task-related component analysis: Tanaka et al
class TRCA(): 
    
    def __init__(self):
        self.W = None
        self.lam = None
        self.Y = None
        self.X = None
    
    def fit(self, X):
        '''
        :param 
        X: data tensor (Nc x Ns x Nt)
        '''
        Nc, Ns, Nt = X.shape

        S = np.zeros((Nc, Nc)) # inter-trial (inter-block) covariance matrix

        # computation of correlation matrices:
        for i in range(Nc):
            for j in range(Nc):
                for k in range(Nt):
                    for l in range(Nt):
                        if k != l: # compare blocks (trials) l and k 
                            xi = X[i, :, k].reshape(1,-1)
                            xj = X[j, :, l].reshape(1,-1)
                            S[i,j] += np.dot((xi-np.mean(xi, axis=1)),(xj-np.mean(xj,axis=1)).T)

        X_bar = X.reshape((Nc, Ns*Nt)) - np.tile(X.reshape((Nc, Ns*Nt)).mean(axis=1).reshape(Nc,1),(1, Ns*Nt))
        
        Q = np.dot(X_bar, X_bar.T) # Nc x Nc data covariance matrix
        lam, W = np.linalg.eig(np.dot(np.linalg.inv(Q), S))
        
        i = np.argsort(np.real(lam))[::-1] # get order of largest eigenvalues in lam
        
        self.X = X
        self.W = W[:, i]
        self.lam = lam[i]
        self.Y = np.dot(self.W[:, 0].T, X_bar)
        
    def compute_corr(self, X_test):
        X_av = self.X.mean(axis=-1)
        w = self.W[:, 0] # get eig. vector corresp to largest eig val
        return pearsonr(np.squeeze(w.T.dot(X_test)), np.squeeze(np.squeeze(w.T.dot(X_av))))[0]
    
    def get_eig(self):
        return self.lam, self.W
    
class TRCA_SSVEP():
    
    def __init__(self, stim_freqs):
        self.stim_freqs = stim_freqs
        self.models = {f: TRCA() for f in stim_freqs} # init independent TRCA models per stim freq

    def fit(self, X_ssvep):
        '''
        Fit the independent Nf TRCA models using input data tensor `X_ssvep`
        
        :param 
        X_ssvep: 4th order data tensor (Nf x Nc x Ns x Nt)
        '''
        assert len(X_ssvep.shape) == 4, "Expected a 4th order data tensor with shape (Nf x Nc x Ns x Nt)"
        assert len(self.stim_freqs) == X_ssvep.shape[0], "Length of supplied stim freqs does not match first dimension of input data"
        
        for i, f in enumerate(self.stim_freqs):
            self.models[f].fit(X_ssvep[i, :, :, :])
    
    def compute_corr(self, X_test):
        assert len(X_test.shape) == 2, "Expected a matrix with shape (Nc x Ns)"
        
        return {f: self.models[f].compute_corr(X_test) for f in self.stim_freqs}
    
    def get_eig(self):
        return {f: self.models[f].get_eig() for f in self.stim_freqs}