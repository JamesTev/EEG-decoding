from ulab import numpy as np
import urandom


def solve_gen_eig_prob(A, B, eps=1e-6):
    """
    Solves the generalised eigenvalue problem of the form:
    Aw = \lambda*Bw

    Note: can be validated against `scipy.linalg.eig(A, b=B)`

    Ref:
    'Eigenvalue and Generalized Eigenvalue Problems: Tutorial (2019)'
    Benyamin Ghojogh and Fakhri Karray and Mark Crowley
    arXiv 1903.11240

    """
    Lam_b, Phi_b = np.linalg.eig(B)  # eig decomp of B alone
    Lam_b = np.eye(len(Lam_b)) * Lam_b  # convert to diagonal matrix of eig vals

    Lam_b_sq = replace_nan(Lam_b ** 0.5) + np.eye(len(Lam_b)) * eps
    Phi_b_hat = np.dot(Phi_b, np.linalg.inv(Lam_b_sq))
    A_hat = np.dot(np.dot(Phi_b_hat.transpose(), A), Phi_b_hat)
    Lam_a, Phi_a = np.linalg.eig(A_hat)
    Lam_a = np.eye(len(Lam_a)) * Lam_a

    Lam = Lam_a
    Phi = np.dot(Phi_b_hat, Phi_a)

    return np.diag(Lam), Phi


def solve_eig_qr(A, iterations=30):

    """
    Use the QR iteration algorithm to iteratively solve for the eigenvectors and eigenvalues
    of a matrix A. Note: only guaranteed to recover exactly for symmetric matrices
    with real eigenvalues. May work partially for asymmetric matrices (no complex support yet).

    Returns:
        `lam`: vector of eigenvalues
        `Q_bar`: matrix of eigenvectors (columns)
    """

    Ak = A
    Q_bar = np.eye(len(Ak))

    for _ in range(iterations):
        Qk, Rk = np.linalg.qr(Ak)
        Ak = np.dot(Rk, Qk)
        Q_bar = np.dot(Q_bar, Qk)

    lam = np.diag(Ak)
    return lam, Q_bar


def power_iteration(A, iterations):
    """
    Iterative algo. to find the eigenvector of a matrix A corresponding to the largest
    eigenvalue.

    TODO: Establish some measure or heuristic of min number of iterations required
    """
    # choose random initial vector to reduce risk of choosing one orthogonal to
    # target eigen vector
    b_k = np.array([urandom.random() for i in range(len(A))])

    for _ in range(iterations):
        b_k1 = np.dot(A, b_k)
        b_k1_norm = np.linalg.norm(b_k1)
        # re normalize the vector
        b_k = b_k1 / b_k1_norm

    return b_k1_norm, b_k


def max_eig(A, iterations, numeric_method="qr"):
    """
    Function to return the largest eigenvalue of a matrix and its corresponding eigenvector.

    A must be square but need not be symmetric. Tries to first use uLab `np.linalg.eig`
    that is better optimised but requires a symmetric matrix. Failing this, power iteration
    algorithm is used.
    """
    try:
        lam, V = np.linalg.eig(A)
        v = V[:, np.argmax(lam)]
    except ValueError:
        if numeric_method == "power_iteration":
            lam, v = power_iteration(A, iterations)
        else:
            if numeric_method != "qr":
                print("Unknown `numeric_method` arg: defaulting to QR solver")
            lam, v = solve_eig_qr(A, 1, lam_iterations=iterations)
            lam = lam[0]  # only need first eigen val (largest returned first)
            v = v[:, 0]  # only first eig vector

    return lam, v


def resample(X, factor):
    idx_rs = np.arange(0, len(X) - 1, factor)
    return X[idx_rs]


def standardise(X):
    axis = np.argmax(X.shape())
    minor_shape = np.min(X.shape())
    mu = np.mean(X, axis=axis).reshape((minor_shape, 1))
    sigma = np.std(X, axis=axis).reshape((minor_shape, 1))
    return (X - mu) / sigma


def cov(X, Y, biased=False):
    assert (
        X.shape() == Y.shape() and len(X.shape()) == 1
    ), "Expected data vectors of equal length"
    assert len(X) > 1, "At least 2 data points are required"

    X = X - np.mean(X)
    Y = Y - np.mean(Y)
    denom = len(X) if biased else len(X) - 1

    return (np.sum(X * Y)) / denom


def corr(X, Y):
    assert (
        X.shape() == Y.shape() and len(X.shape()) == 1
    ), "Expected data vectors of equal length"
    assert len(X) > 1, "At least 2 data points are required"

    return cov(X, Y, biased=True) / (np.std(X) * np.std(Y))


def replace_nan(A, rep=0):
    return np.where(np.isfinite(A), A, rep)
