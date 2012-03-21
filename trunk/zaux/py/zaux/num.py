import numpy as np

acont = np.ascontiguousarray

def V(*v):
    return np.float64(v)

def anorm(v):
    return np.sqrt( (v*v).sum(-1) )