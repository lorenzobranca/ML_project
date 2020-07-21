import jax.numpy as np
from jax import random, grad, vmap, jit
import matplotlib.pyplot as plt
from activation_jax import tanh, sigmoid, leaky_relu
import numpy as slow_np
from metrics import score
from ism_reduced_module import set_IC, loss_ism_reduced, get_coeff_ism_reduced, make_function

#krome---networks/react_galaxy_ism_reduced

def FNN(params,inputs):
        w0 = params[:40]
        b0 = params[40:80]
        w1 = params[80:120]
        b1 = params[120:160]
        w2 = params[160:200]
        b2 = params[200]


        inputs=tanh(inputs*w0+b0)
        
        inputs=tanh(inputs*w1+b1)
        inputs=tanh(np.sum(inputs*w2)+b2)

        return inputs


key = random.PRNGKey(0)
params = random.normal(key, shape=(201,4))

#dfdx = grad(FNN, 1)

#inputs = np.linspace(0., 5., num=300)
inputs=random.uniform(key,shape=(512,),minval=-2.5e0, maxval=7e0)

f_vect1 = vmap(FNN, (None, 0))
#dfdx_vect1 = vmap(dfdx, (None, 0))

f_vect2 = vmap(FNN, (None, 0))
#dfdx_vect2 = vmap(dfdx, (None, 0))

f_vect3 = vmap(FNN, (None, 0))
#dfdx_vect3 = vmap(dfdx, (None, 0))

f_vect4 = vmap(FNN, (None, 0))
#dfdx_vect4 = vmap(dfdx, (None, 0))



Tgas=1e3
flux=1e0



IC=set_IC(FNN,slow_np.array([7.4999999999999997E-002, 8.3333333333333332E-003, 0.83333333333333337,8.3333333333333329E-002]),params,inputs)

k_list=get_coeff_ism_reduced(Tgas,flux)

#f_vect1,f_vect2,f_vect3,f_vect4,dfdx_vect1,dfdx_vect2,dfdx_vect3,dfdx_vect4=make_function(FNN)
#df_list=[dfdx_vect1(params[:,0],inputs),dfdx_vect2(params[:,1],inputs),dfdx_vect3(params[:,2],inputs),dfdx_vect4(params[:,3],inputs)]
#f_list=[f_vect1(params[:,0],inputs),f_vect2(params[:,1],inputs),f_vect3(params[:,2],inputs),f_vect4(params[:,3],inputs)]

loss=loss_ism_reduced(params, inputs,k_list,IC)
print(loss)

grad_loss = jit(grad(loss_ism_reduced, 0))
print(grad_loss)

print('ci siamo')
epochs = 10000

'''
learning_rate = 1e-3
momentum = 0.99
velocity = np.array([0.,0.,0.,0.])
'''
print('parametri settati')

print('daje')
'''
for epoch in range(epochs):
    #if epoch % 50 == 0:
    print('epoch: %3d loss: %.6f' % (epoch, loss_ism_reduced(params, inputs,k_list,IC)))

    gradient = grad_loss(params + momentum*velocity, inputs,k_list,IC)

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
        print('epoch: %3d loss: %.6f' % (epoch, loss_ism_reduced(params, inputs,k_list,IC)))

    gradient = grad_loss(params, inputs,k_list,IC)

    momentum=beta1*momentum+(1e0-beta1)*gradient
    velocity=beta2*velocity+(1e0-beta2)*gradient*gradient

    momentum_norm=momentum/(1e0-beta1**float(epoch+1))
    velocity_norm=velocity/(1e0-beta2**float(epoch+1))

    params-= lr*momentum_norm/(np.sqrt(velocity_norm)+delta)


#print(f_vect1(params[:,0], inputs))
#print(f_vect2(params[:,1], inputs))
#print(f_vect3(params[:,2], inputs))
#print(f_vect4(params[:,3], inputs))


plt.plot(inputs, (f_vect1(params[:,0], inputs)),'.', label='e',color='red')
plt.plot(inputs, (f_vect2(params[:,1], inputs)),'.', label='H-',color='green')
plt.plot(inputs, (f_vect3(params[:,2], inputs)),'.', label='H',color='blue')
plt.plot(inputs, (f_vect4(params[:,3], inputs)),'.', label='H+',color='yellow')
plt.legend()
plt.show()

