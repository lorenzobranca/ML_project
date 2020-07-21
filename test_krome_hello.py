import jax.numpy as np
from jax import random, grad, vmap, jit
import matplotlib.pyplot as plt
from activation_jax import tanh, sigmoid, leaky_relu, relu
import numpy as slow_np
from metrics import score
from jax.config import config
import math


#config.update("jax_debug_nans", True) #debugger for Nan--- kill the code

# krome networks/react_hello---single T---no flux 


def FNN(params,x):
    w0 = params[:20]
    b0 = params[20:40]
    w1 = params[40:60]
    b1 = params[60]

    x=tanh(x*w0+b0)
    x=tanh(np.sum(x*w1)+b1)
    return x

exact=slow_np.loadtxt('NN_jax/fort.66',unpack=True)

key = random.PRNGKey(0)
params = random.normal(key, shape=(61,3))

dfdx = grad(FNN, 1)

inputs = np.linspace(-0., 5., num=1130)
inputs_krome=exact[0,:]

f_vect1 = vmap(FNN, (None, 0))
dfdx_vect1 = vmap(dfdx, (None, 0))

f_vect2 = vmap(FNN, (None, 0))
dfdx_vect2 = vmap(dfdx, (None, 0))

f_vect3 = vmap(FNN, (None, 0))
dfdx_vect3 = vmap(dfdx, (None, 0))
@jit

def loss(params, inputs):
    eq1=dfdx_vect1(params[:,0], inputs)+f_vect1(params[:,0], inputs)
    eq2=dfdx_vect2(params[:,1], inputs)-f_vect1(params[:,0], inputs)+0.5*f_vect2(params[:,1], inputs)
    eq3=dfdx_vect3(params[:,2], inputs)-0.5*f_vect2(params[:,1], inputs)

    ic1=FNN(params[:,0],0.)-1
    ic2=FNN(params[:,1],0.)-1e-10
    ic3=FNN(params[:,2],0.)-1e-10
    
    return sum((np.mean(eq1**2.+ic1**2),np.mean(eq2**2.+ic2**2),np.mean(eq3**2.+ic3**2)))




grad_loss = jit(grad(loss, 0))



'''
learning_rate = 0.0007
momentum = 0.99
velocity = np.array([0.,0.,0.])

for epoch in range(epochs):
    if epoch % 100 == 0:
        print('epoch: %3d loss: %.6f' % (epoch, loss(params, inputs)))
        print(f_vect1(params[:,0],inputs)[0]-1.)
    gradient = grad_loss(params + momentum*velocity, inputs)
    velocity = momentum*velocity - learning_rate*gradient
    params += velocity
'''

lr=8e-3
beta1=9e-1
beta2=999e-3
delta=1e-8
momentum=0e0
velocity=0e0

epoch=0
eps=5e-6

train_loss=[]
validation_loss=[]

#for epoch in range(epochs):
while loss(params,inputs) > eps and epoch < 50000:
    

    if epoch % 200 == 0:
        print('epoch: %3d loss: %.6f' % (epoch, loss(params, inputs)),loss(params,inputs_krome))
    
    train_loss.append(loss(params, inputs))
    validation_loss.append(loss(params,inputs_krome))

    gradient = grad_loss(params, inputs)
    
    momentum=beta1*momentum+(1e0-beta1)*gradient
    velocity=beta2*velocity+(1e0-beta2)*gradient*gradient

    momentum_norm=momentum/(1e0-beta1**float(epoch+1))
    velocity_norm=velocity/(1e0-beta2**float(epoch+1))
    
    epoch=epoch+1

    if slow_np.isnan(momentum_norm).any()==False:
        params-= lr*momentum_norm/(np.sqrt(velocity_norm)+delta)
    else:
        params=params
    
    
train_loss=np.array(train_loss)
validation_loss=np.array(validation_loss)

plt.plot(np.arange(len(train_loss))+1,np.log10(train_loss),label='train')
plt.plot(np.arange(len(train_loss))+1,np.log10(validation_loss),label='valid')
plt.xlabel('epoch')
plt.ylabel('log10[loss]')
plt.legend()
plt.show()
#print(f_vect1(params[:,0], inputs),f_vect2(params[:,1], inputs),f_vect3(params[:,2], inputs))

print('specie 1:')
score(exact[1,:],f_vect1(params[:,0], inputs_krome))

print('specie 2:')
score(exact[2,:],f_vect1(params[:,1], inputs_krome))

print('specie 3:')
score(exact[3,:],f_vect1(params[:,2], inputs_krome))

plt.plot(exact[0,:],exact[1,:],label='A_krome',color='red')
plt.plot(exact[0,:],exact[2,:],label='B_krome',color='green')
plt.plot(exact[0,:],exact[3,:],label='C_krome',color='blue')

plt.plot(inputs_krome, f_vect1(params[:,0], inputs_krome),'.', label='A_pred',color='red')
plt.plot(inputs_krome, f_vect2(params[:,1], inputs_krome),'.', label='B_pred',color='green')
plt.plot(inputs_krome, f_vect3(params[:,2], inputs_krome),'.', label='C_pred',color='blue')
plt.legend()

plt.xlabel('t')
plt.ylabel('n[#/cm3]')
#plt.savefig('krome_hello.png')
plt.show()

