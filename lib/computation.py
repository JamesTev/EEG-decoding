from ulab import numpy as np

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
    Lam_b, Phi_b = np.linalg.eig(B) # eig decomp of B alone
    Lam_b = np.eye(len(Lam_b))*Lam_b # convert to diagonal matrix of eig vals
    
    Lam_b_sq = replace_nan(Lam_b**0.5)+np.eye(len(Lam_b))*eps
    Phi_b_hat = np.dot(Phi_b, np.linalg.inv(Lam_b_sq))
    A_hat = np.dot(np.dot(Phi_b_hat.transpose(), A), Phi_b_hat)
    Lam_a, Phi_a = np.linalg.eig(A_hat)
    Lam_a = np.eye(len(Lam_a))*Lam_a
    
    Lam = Lam_a
    Phi = np.dot(Phi_b_hat, Phi_a)
    
    return np.diag(Lam), Phi

def resample(X, factor):
    idx_rs = np.arange(0, len(X)-1, factor)
    return X[idx_rs]

def standardise(X):
    axis = np.argmax(X.shape())
    minor_shape = np.min(X.shape())
    mu = np.mean(X, axis=axis).reshape((minor_shape, 1))
    sigma = np.std(X, axis=axis).reshape((minor_shape, 1))
    return (X-mu)/sigma

def cov(X, Y, biased=False):
    assert X.shape() == Y.shape() and len(X.shape()) == 1, "Expected data vectors of equal length"
    assert len(X) > 1, "At least 2 data points are required"
    
    X = X - np.mean(X)
    Y = Y - np.mean(Y)
    denom = len(X) if biased else len(X)-1
    
    return (np.sum(X*Y))/denom

def corr(X, Y):
    assert X.shape() == Y.shape() and len(X.shape()) == 1, "Expected data vectors of equal length"
    assert len(X) > 1, "At least 2 data points are required"
    
    return cov(X,Y, biased=True)/(np.std(X)*np.std(Y))

def replace_nan(A, rep=0):
    return np.where(np.isfinite(A), A, rep)