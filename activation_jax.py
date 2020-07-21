#activation function personal library

import jax.numpy as np

#hyperbolic

def tanh(x):

    y=np.exp(2.*x)
    return (y-1.)/(y+1.)

def sigmoid(x):
    return 1./(1. + np.exp(-x))

#linear

def relu(x):

    return np.max(0.,x)

def leaky_relu(x,alpha=0.01):

    y=np.zeros_like(x)

    result=np.where(x>0,x,y)

    return result



