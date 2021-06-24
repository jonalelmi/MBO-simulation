#!/bin/python
import numpy as np
from scipy.linalg import expm
def MBO(L, u, h, mz = False):
    
    exp_Lh = expm(-h*L)
    
    u_h = np.matmul(exp_Lh, u)
    
    if mz == True: #mean 1/2 constrain
        u_h = u_h - np.mean(u_h) + 1/2
        
    u_new = np.heaviside(u_h - 1/2, 1)
    
    return u_new


