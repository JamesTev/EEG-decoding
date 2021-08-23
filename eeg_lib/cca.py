from sklearn.cross_decomposition import CCA as CCA_sklearn
from .filtering import filterbank
import numpy as np

from scipy.stats import pearsonr
from scipy.linalg import block_diag

from .utils import resample, standardise, solve_gen_eig_prob

np.random.seed(0)


class GCCA():
    """
    Generalised canonical component analysis.
    
    Ref: 'Improving SSVEP Identification Accuracy via Generalized Canonical Correlation Analysis'
    Sun, Chen et al
    """
    
    def __init__(self, f_ssvep, fs, Nh=3, w=None, name=None):
        self.Nc, self.Ns, self.Nt = None, None, None
        self.Nh = Nh
        self.w_chi_bar_n = None
        self.w_Y_n = None
        self.w_Chi_n = None
        self.fs = fs
        self.f_ssvep = f_ssvep
        
        self.name = name or "gcca_{0}hz".format(f_ssvep)
            
        
    def fit(self, X):
        """
        Fit against training tensor X.
        
        X should be a 3rd order tensor of dim (Nc x Ns x Nt)
        """
        assert len(X.shape) == 3, "Expected 4th order input data tensor: Nc x Ns x Nt"
        self.Nc, self.Ns, self.Nt = X.shape
        
        Chi_n = X
        Chi_n_c = Chi_n.reshape((self.Nc, self.Ns*self.Nt))

        Chi_bar_n = np.mean(Chi_n, axis=-1) # mean over trials for each channel with all samples: output shape is Nc x Ns x 1
        Chi_bar_n_c = np.concatenate([Chi_bar_n for i in range(self.Nt)], axis=1) # concat along columns

        Y_n = cca_reference([self.f_ssvep], self.fs, self.Ns, Nh=self.Nh).reshape(-1, self.Ns)
        Y_n_c = np.concatenate([Y_n for i in range(self.Nt)], axis=1)

        # form X and D and find eigenvals
        X = np.c_[Chi_n_c.T, Chi_bar_n_c.T, Y_n_c.T].T

        d1 = Chi_n_c.dot(Chi_n_c.T)
        d2 = Chi_bar_n_c.dot(Chi_bar_n_c.T)
        d3 = Y_n_c.dot(Y_n_c.T)
        D = block_diag(d1, d2, d3)

        lam, W_eig = solve_gen_eig_prob(X.dot(X.T), D) # solve generalised eigenvalue problem

        i = np.argmax(np.real(lam))
        w = W_eig[:, i] # optimal spatial filter vector with dim (2*Nc + 2*Nh)
        
        w_Chi_n = w[:self.Nc] # first Nc weight values correspond to data channels
        w_Chi_bar_n = w[self.Nc:2*self.Nc] # second Nc weights correspond to Nc template channels
        w_Y_n = w[2*self.Nc:] # final 2*Nh weights correspond to ref sinusoids with harmonics
        
        self.w_chi_bar_n =  w_Chi_bar_n.T.dot(Chi_bar_n)
        self.w_Y_n = w_Y_n.T.dot(Y_n)
        self.w_Chi_n = w_Chi_n
        
            
    def classify(self, X_test):
        if self.w_chi_bar_n is None:
            raise ValueError("call `.fit(X_train)` before performing classification.")

        rho1 = pearsonr(self.w_Chi_n.T.dot(X_test), self.w_chi_bar_n)[0]
        rho2 = pearsonr(self.w_Chi_n.T.dot(X_test), self.w_Y_n)[0]

        rho = np.sum([np.sign(rho_i)*rho_i**2 for rho_i in [rho1, rho2]])
        
        return rho

class GCCA_SSVEP():
    """
    Generalised canonical component analysis.
    
    Ref: 'Improving SSVEP Identification Accuracy via Generalized Canonical Correlation Analysis'
    Sun, Chen et al
    """
    
    def __init__(self, stim_freqs, fs, Nh=3, W=None):
        self.Nf, self.Nc, self.Ns, self.Nt = None, None, None, None
        self.Nh = Nh
        self.W = W
        self.stim_freqs = stim_freqs
        self.fs = fs
        
    def fit(self, X):
        """
        Fit against training tensor X.
        
        X should be a 4th order tensor of dim (Nf x Nc x Ns x Nt)
        """
        assert len(X.shape) == 4, "Expected 4th order input data tensor: Nf x Nc x Ns x Nt"
        self.Chi = X
        self.Nf, self.Nc, self.Ns, self.Nt = X.shape
        
        W = []
        self.Chi_bar = []
        self.Y = []
        for n in range(len(self.stim_freqs)):
            Chi_n = self.Chi[n, :, :, :]
            Chi_n_c = Chi_n.reshape((self.Nc, self.Ns*self.Nt))

            Chi_bar_n = np.mean(Chi_n, axis=-1) # mean over trials for each channel with all samples: output shape is Nc x Ns x 1
            self.Chi_bar.append(Chi_bar_n)
            Chi_bar_n_c = np.concatenate([Chi_bar_n for i in range(self.Nt)], axis=1) # concat along columns

            Y_n = cca_reference([self.stim_freqs[n]], self.fs, self.Ns, Nh=self.Nh).reshape(-1, self.Ns)
            self.Y.append(Y_n)
            Y_n_c = np.concatenate([Y_n for i in range(self.Nt)], axis=1)

            # form X and D and find eigenvals
            X = np.c_[Chi_n_c.T, Chi_bar_n_c.T, Y_n_c.T].T

            d1 = Chi_n_c.dot(Chi_n_c.T)
            d2 = Chi_bar_n_c.dot(Chi_bar_n_c.T)
            d3 = Y_n_c.dot(Y_n_c.T)
            D = block_diag(d1, d2, d3)

            lam, W_eig = solve_gen_eig_prob(X.dot(X.T), D) # solve generalised eigenvalue problem

            i = np.argmax(np.real(lam))
            W.append(W_eig[:, i]) # optimal spatial filter vector with dim (2*Nc + 2*Nh)
        
        self.Chi_bar = np.array(self.Chi_bar) # form tensors 
        self.Y = np.array(self.Y)
        self.W = np.array(W)
            
    def classify(self, X_test):
        if self.W is None:
            raise ValueError("w must be computed using `compute_w` before performing classification.")
        result = {f:0 for f in self.stim_freqs}
        
        for i in range(len(self.stim_freqs)):
            Chi_bar_n = self.Chi_bar[i, :, :]
            Y_n = self.Y[i, :, :]
            
            w = self.W[i, :]
            w_Chi_n = w[:self.Nc] # first Nc weight values correspond to data channels
            w_Chi_bar_n = w[self.Nc:2*self.Nc] # second Nc weights correspond to Nc template channels
            w_Y_n = w[2*self.Nc:] # final 2*Nh weights correspond to ref sinusoids with harmonics

            rho1 = pearsonr(w_Chi_n.T.dot(X_test), w_Chi_bar_n.T.dot(Chi_bar_n))[0]
            rho2 = pearsonr(w_Chi_n.T.dot(X_test), w_Y_n.T.dot(Y_n))[0]

            rho_n = sum([np.sign(rho_i)*rho_i**2 for rho_i in [rho1, rho2]])
            result[self.stim_freqs[i]] = rho_n
        return result
    
class MsetCCA():
    """
    Multi set CCA
    
    Ref: FREQUENCY RECOGNITION IN SSVEP-BASED BCI USING MULTISET CANONICAL CORRELATION ANALYSIS, Zhang, Zhou et al
    """
    
    def __init__(self):
        self.Nc, self.Ns, self.Nt = None, None, None
        
    def fit(self, chi):
        """
        Fit against training tensor chi.
        
        chi should be a 3rd order tensor of dim (Nc x Ns x Nt)
        """
        assert len(chi.shape) == 3, "Expected 3rd order input data tensor for freq. fm: Nc x Ns x Nt"
        
        Nc, Ns, Nt = chi.shape
        
        chi_c = np.vstack([chi[:, :, i] for i in range(Nt)])
        R = chi_c.dot(chi_c.T)

        # form inra-trial covariance matrix S
        blocks = [chi[:, :, i].dot(chi[:, :, i].T) for i in range(Nt)]
        S = block_diag(*blocks)

        lam, V = solve_gen_eig_prob((R-S), S) # solve generalise eig value problem
        w = V[:, np.argmax(lam)].reshape((Nt, Nc)) # sort by largest eig vals in lam vector. TODO: check reshaping

        self.Y = np.array([w[i, :].T.dot(chi[:, :, i]) for i in range(Nt)]) # form optimised reference matrix
        self.Nc, self.Ns, self.Nt = Nc, Ns, Nt
            
    def compute_corr(self, X_test, method='cca'):
        if self.Y is None:
            raise ValueError("Reference matrix Y must be computed using `fit` before computing corr")
        if method == 'eig':
            rho = CCA.cca_eig(X_test, self.Y)[0]
        else: # use sklearn implementation
            cca = CCA_sklearn(n_components=1)
            Xc, Yc = cca.fit_transform(X_test.T, self.Y.T)
            rho = pearsonr(Xc[:, 0], Yc[:, 0])[0]
        return rho

class MsetCCA_SSVEP():
    
    def __init__(self, stim_freqs):
        self.stim_freqs = stim_freqs
        self.models = {f: MsetCCA() for f in stim_freqs} # init independent TRCA models per stim freq

    def fit(self, X_ssvep):
        '''
        Fit the independent Nf MsetCCA models using input data tensor `X_ssvep`
        
        :param 
        X_ssvep: 4th order data tensor (Nf x Nc x Ns x Nt)
        '''
        assert len(X_ssvep.shape) == 4, "Expected a 4th order data tensor with shape (Nf x Nc x Ns x Nt)"
        assert len(self.stim_freqs) == X_ssvep.shape[0], "Length of supplied stim freqs does not match first dimension of input data"
        
        for i, f in enumerate(self.stim_freqs):
            self.models[f].fit(X_ssvep[i, :, :, :])
    
    def classify(self, X_test, method='cca'):
        assert len(X_test.shape) == 2, "Expected a matrix with shape (Nc x Ns)"
        
        return {f: self.models[f].compute_corr(X_test, method=method) for f in self.stim_freqs}    
class CCA():
    
    def __init__(self, stim_freqs, fs, Nh=2):
        self.Nh = Nh
        self.stim_freqs = stim_freqs
        self.fs = fs
        
    def compute_corr(self, X_test):            
        result = {}
        Cxx = np.dot(X_test, X_test.transpose()) # precompute data auto correlation matrix
        for f in self.stim_freqs:
            Y = cca_reference([f], self.fs, np.max(X_test.shape), Nh=self.Nh, standardise_out=False)
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

def fbcca(eeg, list_freqs, fs, num_harms=3, num_fbs=5):
    
    """
    Steady-state visual evoked potentials (SSVEPs) detection using the filter
    bank canonical correlation analysis (FBCCA)-based method [1].
    function results = test_fbcca(eeg, list_freqs, fs, num_harms, num_fbs)
    Input:
      eeg             : Input eeg data 
                        (# of targets, # of channels, Data length [sample])
      list_freqs      : List for stimulus frequencies
      fs              : Sampling frequency
      num_harms       : # of harmonics
      num_fbs         : # of filters in filterbank analysis
    Output:
      results         : The target estimated by this method
    Reference:
      [1] X. Chen, Y. Wang, S. Gao, T. -P. Jung and X. Gao,
          "Filter bank canonical correlation analysis for implementing a 
           high-speed SSVEP-based brain-computer interface",
          J. Neural Eng., vol.12, 046008, 2015.
    """
    
    fb_coefs = np.power(np.arange(1,num_fbs+1),(-1.25)) + 0.25
    
    num_targs, _, num_smpls = eeg.shape  #40 taget (means 40 fre-phase combination that we want to predict)
    y_ref = cca_reference(list_freqs, fs, num_smpls, num_harms)
    cca = CCA(n_components=1) #initilize CCA
    
    # result matrix
    r = np.zeros((num_fbs,num_targs))
    results = np.zeros(num_targs)
    
    for targ_i in range(num_targs):
        test_tmp = np.squeeze(eeg[targ_i, :, :])  #deal with one target a time
        for fb_i in range(num_fbs):  #filter bank number, deal with different filter bank
            testdata = filterbank(test_tmp, fs, fb_i)  #data after filtering
            for class_i in range(num_targs):
                refdata = np.squeeze(y_ref[class_i, :, :])   #pick corresponding freq target reference signal
                test_C, ref_C = cca.fit_transform(testdata.T, refdata.T)
                # len(row) = len(observation), len(column) = variables of each observation
                # number of rows should be the same, so need transpose here
                # output is the highest correlation linear combination of two sets
                r_tmp, _ = pearsonr(np.squeeze(test_C), np.squeeze(ref_C)) #return r and p_value, use np.squeeze to adapt the API 
                r[fb_i, class_i] = r_tmp
                 
        rho = np.dot(fb_coefs, r)  #weighted sum of r from all different filter banks' result
        tau = np.argmax(rho)  #get maximum from the target as the final predict (get the index)
        results[targ_i] = tau #index indicate the maximum(most possible) target
    return results
    
def cca_reference(list_freqs, fs, Ns, Nh=3, standardise_out=False):
    
    '''
    Generate reference signals for the canonical correlation analysis (CCA)
    -based steady-state visual evoked potentials (SSVEPs) detection [1, 2].
    function [ y_ref ] = cca_reference(listFreq, fs,  Ns, Nh) 
    Input:
      list_freqs        : stimulus frequencies
      fs              : Sampling frequency
      Ns              : # of samples in trial
      Nh          : # of harmonics
    Output:
      y_ref           : Generated reference signals with shape (Nf, Ns, 2*Nh)
    '''  

    num_freqs = len(list_freqs)
    tidx = np.arange(1,Ns+1)/fs #time index
    
    y_ref = np.zeros((num_freqs, 2*Nh, Ns))
    for freq_i in range(num_freqs):
        tmp = []
        for harm_i in range(1,Nh+1):
            stim_freq = list_freqs[freq_i]  #in HZ
            # Sin and Cos
            tmp.extend([np.sin(2*np.pi*tidx*harm_i*stim_freq),
                       np.cos(2*np.pi*tidx*harm_i*stim_freq)])
        y_ref[freq_i] = tmp # 2*num_harms because include both sin and cos
    
    y_ref = np.squeeze(y_ref)
    if standardise_out: # zero mean, unit std. dev
        return standardise(y_ref)
    return y_ref

'''
Base on fbcca, but adapt to our input format
'''   
def fbcca_realtime(data, list_freqs, fs, num_harms=3, num_fbs=5):
    
    fb_coefs = np.power(np.arange(1,num_fbs+1),(-1.25)) + 0.25
    
    num_targs = len(list_freqs)
    _, num_smpls = data.shape
    
    y_ref = cca_reference(list_freqs, fs, num_smpls, num_harms)
    cca = CCA(n_components=1) #initialize CCA
    
    # result matrix
    r = np.zeros((num_fbs,num_targs))
    
    for fb_i in range(num_fbs):  #filter bank number, deal with different filter bank
        testdata = filterbank(data, fs, fb_i)  #data after filtering
        for class_i in range(num_targs):
            refdata = np.squeeze(y_ref[class_i, :, :])   #pick corresponding freq target reference signal
            test_C, ref_C = cca.fit_transform(testdata.T, refdata.T)
            r_tmp, _ = pearsonr(np.squeeze(test_C), np.squeeze(ref_C)) #return r and p_value
            if r_tmp == np.nan:
                r_tmp=0
            r[fb_i, class_i] = r_tmp
    
    rho = np.dot(fb_coefs, r)  #weighted sum of r from all different filter banks' result
    print(rho) #print out the correlation
    result = np.argmax(rho)  #get maximum from the target as the final predict (get the index), and index indicates the maximum entry(most possible target)
    ''' Threshold '''
    THRESHOLD = 2.1
    if abs(rho[result])<THRESHOLD:  #2.587=np.sum(fb_coefs*0.8) #2.91=np.sum(fb_coefs*0.9) #1.941=np.sum(fb_coefs*0.6)
        return 999 #if the correlation isn't big enough, do not return any command
    else:
        return result
    
def standardise(X):
    axis = np.argmax(X.shape())
    minor_shape = np.min(X.shape())
    mu = np.mean(X, axis=axis).reshape((minor_shape, 1))
    sigma = np.std(X, axis=axis).reshape((minor_shape, 1))
    return (X-mu)/sigma

def solve_eig_qr(A, iterations=30):

    Ak = A
    Q_bar = np.eye(len(Ak))

    for _ in range(iterations):
        Qk, Rk = np.linalg.qr(Ak)
        Ak = np.dot(Rk, Qk)
        Q_bar = np.dot(Q_bar, Qk)

    lam = np.diag(Ak)
    return lam, Q_bar