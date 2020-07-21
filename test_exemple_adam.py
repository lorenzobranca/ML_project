import jax.numpy as np
from activation_jax import tanh

def sigmoid(x):
    return 1./(1. + np.exp(-x))

def f(params, x):
    w0 = params[:10]
    b0 = params[10:20]
    w1 = params[20:30]
    b1 = params[30]
    
    x = sigmoid(x*w0 + b0)
    x = sigmoid(np.sum(x*w1) + b1)
    
    return x

from jax import random

key = random.PRNGKey(0)
params = random.normal(key, shape=(31,))

from jax import grad

dfdx = grad(f, 1)

inputs = np.linspace(-2., 2., num=401)

from jax import vmap

f_vect = vmap(f, (None, 0))
dfdx_vect = vmap(dfdx, (None, 0))


from jax import jit

@jit
def loss(params, inputs):
    eq = dfdx_vect(params, inputs) + 2.*inputs*f_vect(params, inputs)
    print(inputs) 
    ic = f(params, 0.) - 1.
    return np.mean(eq**2) + ic**2

grad_loss = jit(grad(loss, 0))



epochs = 7000
'''
learning_rate = 0.05
momentum = 0.99
velocity = 0.

for epoch in range(epochs):
    if epoch % 100 == 0:
        print('epoch: %3d loss: %.6f' % (epoch, loss(params, inputs)))
    gradient = grad_loss(params + momentum*velocity, inputs)
    
    velocity = momentum*velocity - learning_rate*gradient
    params += velocity
'''

lr=1e-2
beta1=9e-1
beta2=999e-3
delta=1e-8
momentum=0e0
velocity=0e0

for epoch in range(epochs):

    if epoch % 100 == 0:
        print('epoch: %3d loss: %.6f' % (epoch, loss(params, inputs)))
    
    gradient = grad_loss(params, inputs)

    momentum=beta1*momentum+(1e0-beta1)*gradient
    velocity=beta2*velocity+(1e0-beta2)*gradient*gradient

    momentum_norm=momentum/(1e0-beta1**float(epoch+1))
    velocity_norm=velocity/(1e0-beta2**float(epoch+1))
    
    params-= lr*momentum_norm/(np.sqrt(velocity_norm)+delta)

import matplotlib.pyplot as plt

plt.plot(inputs, np.exp(-inputs**2), label='exact')
plt.plot(inputs, f_vect(params, inputs), label='approx')
plt.legend()
plt.show()
