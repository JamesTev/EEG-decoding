from ulab import numpy as np
import gc

from .computation import solve_eig_qr, solve_gen_eig_prob, standardise, col_concat, corr, block_diag, sign

class SingleChannelMsetCCA(): 
    """
    Multiset CCA algorithm for SSVEP decoding.
    Computes optimised reference signal set based on historical observations
    and uses ordinary CCA for final correlation computation given a new test
    signal.
    Note: this is a 1 channel implementation (Nc=1)
    """
    def __init__(self):
        self.Ns, self.Nt = None, None
        
    def fit(self, X, compress_ref=True): 
        """
        Expects a training matrix X of shape Nt x Ns. If `compress_ref=True`, the `Nt` components in optimised reference signal Y will be averaged to form a single reference vector. This can be used for memory optimisation but will likely degrade performance slightly.
        """
        if X.shape[0] > X.shape[1]:
            print("Warning: received more trials than samples. This is unusual behaviour: check X")
        
        R = np.dot(X, X.transpose()) # inter trial covariance matrix
        S = np.eye(len(R))*np.diag(R) # intra-trial diag covariance matrix
        lam, V = solve_gen_eig_prob((R-S), S) # solve generalised eig problem
        w = V[:, np.argmax(lam)] # find eigenvector corresp to largest eigenvalue
        Y = np.array([x*w[i] for i, x in enumerate(X)]) # store optimised reference vector Nt x Ns self.Y = Y
        
        if compress_ref:
            self.Y = np.mean(Y, axis=0).reshape((1, max(Y.shape))) # this will average Nt components in Y: Nc x Nt -> 1 x Nt
    
    def compute(self, X_test):
        if self.Y is None:
            raise ValueError("Reference matrix Y must be computed using  fit  before computing corr")
        if len(X_test.shape) == 1:
            X_test = X_test.reshape((1, len(X_test)))
        return CCA.cca_eig(X_test, self.Y)[0] # use ordinary CCA with optimised ref. Y

class SingleChannelGCCA(): 
    """
    Generalised canonical component analysis for Nc=1.
    Expects the target frequency at `f_ssvep`. `fs` is the sampling rate used and `Nh` the number of harmonics for the harmonic r
    Ref: 'Improving SSVEP Identification Accuracy via Generalized Canonical Correlation Analysis' Sun, Chen et al
    """
    def __init__(self, f_ssvep, fs, Nh=1, name=None):
        self.Ns, self.Nt = None, None
        self.Nh = Nh
        self.w = None
        self.X_bar = None
        self.fs = fs
        self.f_ssvep = f_ssvep
        self.name = name or "gcca_{0}hz".format(f_ssvep)
    
    def fit(self, X): 
        """
        Fit against training data.
        X should be a matrix of dim (Nt x Ns)
        """
        self.Nt, self.Ns = X.shape

        # template signal
        X_bar = np.mean(X, axis=0).reshape((1, self.Ns))
        Y = harmonic_reference(self.f_ssvep, self.fs, self.Ns)

        # form concatenated matrices (vectors for Nc=1)
        X_c = X.reshape((1, self.Ns*self.Nt))
        
        X_bar_c = col_concat(*[X_bar for i in range(self.Nt)])
        X_bar_c = X_bar_c.reshape((1, self.Ns*self.Nt))
        
        Y_c = col_concat(*[Y for i in range(self.Nt)])
        
        X_comb = col_concat(X_c.T, X_bar_c.T, Y_c.T).T
        
        D1 = np.dot(X_c, X_c.T)
        D2 = np.dot(X_bar_c, X_bar_c.T)
        D3 = np.dot(Y_c, Y_c.T)
        
        D = block_diag(block_diag(D1, D2), D3)
        
        lam, W_eig = solve_gen_eig_prob(np.dot(X_comb, X_comb.T), D)

        self.w = W_eig[:, np.argmax(lam)] # optimal spatial filter vector with dim (2*Nc + 2*Nh)
        self.X_bar = X_bar
        
    def compute(self, X_test):
        """
        Compute output correlation for a test observation with dim. (1 x Ns)
        """
        if self.w is None:
            raise ValueError("call .fit(X_train) before performing classification.")
            
        if len(X_test.shape) == 1:
            X_test = X_test.reshape((len(X_test), 1))
        else:
            X_test = X_test.T 

        w_X = self.w[0:1]
        w_X_bar = self.w[1:2] # second weight correspond to Nc (Nc=1) template channels
        w_Y = self.w[2:] # final 2*Nh weights correspond to ref sinusoids with harmonics

        # regenerate these instead of storing from the `fit` function since
        # computationally cheap to generate but expensive to store in memory
        Y = harmonic_reference(self.f_ssvep, self.fs, self.Ns)

        X_test_image = np.dot(X_test, w_X)
        rho1 = corr(X_test_image, np.dot(self.X_bar.T, w_X_bar))
        rho2 = corr(X_test_image, np.dot(Y.T, w_Y))
        
        return sum([sign(rho_i)*rho_i**2 for rho_i in [rho1, rho2]])/2
class CCA:
    def __init__(self, stim_freqs, fs, Nh=2):
        self.Nh = Nh
        self.stim_freqs = stim_freqs
        self.fs = fs

    def compute_corr(self, X_test):
        result = {}
        Cxx = np.dot(
            X_test, X_test.transpose()
        )  # precompute data auto correlation matrix
        for f in self.stim_freqs:
            Y = harmonic_reference(
                f, self.fs, np.max(X_test.shape), Nh=self.Nh, standardise_out=False
            )
            rho = self.cca_eig(
                X_test, Y, Cxx=Cxx
            )  # canonical variable matrices. Xc = X^T.W_x
            result[f] = rho
        return result

    @staticmethod
    def cca_eig(X, Y, Cxx=None, eps=1e-6):
        if Cxx is None:
            Cxx = np.dot(X, X.transpose())  # auto correlation matrix
        Cyy = np.dot(Y, Y.transpose())
        Cxy = np.dot(X, Y.transpose())  # cross correlation matrix
        Cyx = np.dot(Y, X.transpose())  # same as Cxy.T

        M1 = np.dot(np.linalg.inv(Cxx + eps), Cxy)  # intermediate result
        M2 = np.dot(np.linalg.inv(Cyy + eps), Cyx)

        lam, _ = solve_eig_qr(np.dot(M1, M2), 20)
        return np.sqrt(lam)


def harmonic_reference(f0, fs, Ns, Nh=1, standardise_out=False):

    """
    Generate reference signals for canonical correlation analysis (CCA)
    -based steady-state visual evoked potentials (SSVEPs) detection [1, 2].
    function [ y_ref ] = cca_reference(listFreq, fs,  Ns, Nh)
    Input:
      f0        : stimulus frequency
      fs              : Sampling frequency
      Ns              : # of samples in trial
      Nh          : # of harmonics
    Output:
      X           : Generated reference signals with shape (Nf, Ns, 2*Nh)
    """
    X = np.zeros((Nh * 2, Ns))

    for harm_i in range(Nh):
        # Sin and Cos
        X[2 * harm_i, :] = np.sin(
            np.arange(1, Ns + 1) * (1 / fs) * 2 * np.pi * (harm_i + 1) * f0
        )
        gc.collect()
        X[2 * harm_i + 1, :] = np.cos(
            np.arange(1, Ns + 1) * (1 / fs) * 2 * np.pi * (harm_i + 1) * f0
        )
        gc.collect()

    # print(micropython.mem_info(1))
    if standardise_out:  # zero mean, unit std. dev
        return standardise(X)
    return X
