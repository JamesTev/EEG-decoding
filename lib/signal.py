from ulab import numpy as np
#     import urandom

# X correlation matrix
Cxx = np.array([[166.14279703,  99.48821575,  24.11370662, -33.37944544],
       [ 99.48821575, 166.12703417,  90.66107192,  71.18545196],
       [ 24.11370662,  90.66107192, 166.83747439,  66.71419887],
       [-33.37944544,  71.18545196,  66.71419887, 166.71542984]])

# Y correlation matrix
Cyy = np.array([[167.        ,   0.43605292,   1.06028619,  -0.1970634 ],
       [  0.43605292, 167.        ,   0.41206453,  -0.69571169],
       [  1.06028619,   0.41206453, 167.        ,  -5.19137319],
       [ -0.1970634 ,  -0.69571169,  -5.19137319, 167.        ]])

# XY cross-correlation matrix
Cxy = np.array([[18.49359001, -0.09579883, -1.47948752,  5.81688046],
       [ 9.60223537,  7.45818488, -1.80016572,  7.20337205],
       [17.57176942, -8.21102309, -0.50311062,  1.26140711],
       [-4.71848465, 15.03455974, -0.45756603,  1.86271787]])

def dummy_mat(dim=4):
    return np.arange(dim**2).reshape((dim, dim))

def dummy_signal(a=1, f=10, noise_pow=0.1, n=10):
    return [a*np.sin(x*f*np.pi*2) for x in np.arange(10)]

def dummy_calc():
    global Cxx, Cxy, Cyy
    M1 = np.dot(np.linalg.inv(Cxx), Cxy)
    M2 = np.dot(np.linalg.inv(Cyy), Cxy.transpose())
    return np.dot(M1, M2)