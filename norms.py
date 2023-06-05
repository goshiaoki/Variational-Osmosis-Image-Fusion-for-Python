import numpy as np

def norms(z, p, dir):
    if p == 1:
        y = np.sum(np.abs(z), dir)
    elif p == np.inf:
        y = np.max(z)
    else:
        y = np.power(np.sum(np.power(z, p), dir), 1/p)
    return y
