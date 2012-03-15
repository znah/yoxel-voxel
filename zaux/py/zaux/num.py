import numpy as np

def V(*v):
    return np.float64(v)

def anorm(v):
    return np.sqrt( (v*v).sum(-1) )