
from sklearn.cross_decomposition import CCA as CCA_sklearn
from .filtering import filterbank
import numpy as np

from scipy.stats import pearsonr
from scipy.linalg import block_diag

from .utils import resample, standardise

np.random.seed(0)

class GCCA():
    """
    Generalised canonical component analysis.
    
    Ref: 'Improving SSVEP Identification Accuracy via Generalized Canonical Correlation Analysis'
    Sun, Chen et al
    """
    
    def __init__(self, data, stim_freqs, fs, Nh=3, W=None):
        assert len(data.shape) == 4, "Expected 4th order input data tensor: Nf x Nc x Ns x Nt"
        self.Nf, self.Nc, self.Ns, self.Nt = data.shape
        self.Nh = Nh
        self.Chi = data
        self.W = W
        self.stim_freqs = stim_freqs
        self.fs = fs
        
    def fit(self):
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

            A = np.linalg.inv(D).dot(X.dot(X.T))
            lam, W_eig = np.linalg.eig(A)

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
    
class CCA():
    
    def __init__(self, stim_freqs, fs, Nh=3):
        self.Nh = Nh
        self.stim_freqs = stim_freqs
        self.fs = fs
        self.cca_models =  {f:CCA_sklearn(n_components=1) for f in stim_freqs}
        self.is_fit = False
        
    def fit(self, X, resampling_factor=None):
        for f in self.stim_freqs:
            Y = cca_reference([f], self.fs, len(X), Nh=self.Nh, standardise_out=True)

            if resampling_factor is not None:
                X = resample(X, resampling_factor)

            self.cca_models[f].fit(X, Y)
            
        self.is_fit = True
        
    def classify(self, X_test, method='eig'):
        if not self.is_fit and method != 'eig':
            self.fit(X_test)
            
        result = {}
        for f in self.stim_freqs:
            Y = cca_reference([f], self.fs, len(X_test), Nh=self.Nh, standardise_out=True)
            if method == 'eig':
                rho = self.cca_eig(X_test, Y)[0]
            else:
                Xc, Yc = self.cca_models[f].transform(X_test, Y) # canonical variable matrices. Xc = X^T.W_x
                rho = pearsonr(Xc[:, 0], Yc[:, 0])[0]
            result[f] = rho
        return result
    
    @staticmethod
    def cca_eig(X, Y, n_components=1):
        Cxx = X.T.dot(X) # auto correlation matrix
        Cyy = Y.T.dot(Y) 
        Cxy = X.T.dot(Y) # cross correlation matrix
        Cyx = Y.T.dot(X) # same as Cxy.T

        M1 = np.linalg.inv(Cxx).dot(Cxy) # intermediate result
        M2 = np.linalg.inv(Cyy).dot(Cyx)

        M = M1.dot(M2)
        lam = np.linalg.eigvals(M)
        return sorted(np.sqrt(lam), reverse=True)[:n_components] # return largest n sqrt eig vals

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
    
    y_ref = np.squeeze(y_ref).T
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