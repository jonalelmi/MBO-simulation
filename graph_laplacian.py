#!/bin/python
import numpy as np

def graph_Laplacian(vertices, similarity_func, eps = 1, lap = "RW"):
    d = vertices.shape[1]
    n = vertices.shape[0]
    
    W = np.zeros((n,n))
    deg = np.zeros(n)
    
    for i in range(n):
        for j in range(n):
            W[i,j] = (1/(eps**d))*similarity_func(np.linalg.norm(vertices[i,:]-vertices[j,:])**2/(eps**2))
    
    deg = 1/n * np.sum(W, axis = 1) #degrees
    D_inv = np.diag(1/deg)
    
    if lap == "RW":
        L = (1/(eps**2))*(np.eye(n) - np.matmul(D_inv, W/n))
    else:
        if lap == "UN":
            D = np.diag(deg)
            L = (1/(eps**2))*(D - W/n)
        else:
            #symmetric laplacian
            DS = np.diag(deg**(-1/2))
            L = (1/(eps**2))*(np.eye(n) - np.matmul(DS, np.matmul(W/n, DS)))
    
    return L, W, deg


