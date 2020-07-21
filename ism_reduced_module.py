import jax.numpy as np
import numpy as slow_np
from jax import jit, grad, vmap
from activation_jax import tanh, sigmoid, leaky_relu
#4 species---10 reaction



def set_IC(FNN,IC_list,params,inputs):
    
    IC=slow_np.zeros(len(IC_list))
    print(IC_list)
    for i in range(len(IC_list)):

        IC[i]=FNN(params[:,i],0.)-IC_list[i]
    print(IC)    

    return IC

def make_function(FNN):

    dfdx = grad(FNN, 1)

    f_vect1 = vmap(FNN, (None, 0))
    dfdx_vect1 = vmap(dfdx, (None, 0))

    f_vect2 = vmap(FNN, (None, 0))
    dfdx_vect2 = vmap(dfdx, (None, 0))

    f_vect3 = vmap(FNN, (None, 0))
    dfdx_vect3 = vmap(dfdx, (None, 0))

    f_vect4 = vmap(FNN, (None, 0))
    dfdx_vect4 = vmap(dfdx, (None, 0))

    return f_vect1,f_vect2,f_vect3,f_vect4,dfdx_vect1,dfdx_vect2,dfdx_vect3,dfdx_vect4

@jit
def loss_ism_reduced(params, inputs,k,IC):

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




    #def make_function(FNN):

    dfdx = grad(FNN, 1)

    f_vect1 = vmap(FNN, (None, 0))
    dfdx_vect1 = vmap(dfdx, (None, 0))
        
    f_vect2 = vmap(FNN, (None, 0))
    dfdx_vect2 = vmap(dfdx, (None, 0))

    f_vect3 = vmap(FNN, (None, 0))
    dfdx_vect3 = vmap(dfdx, (None, 0))

    f_vect4 = vmap(FNN, (None, 0))
    dfdx_vect4 = vmap(dfdx, (None, 0))

        #return f_vect1,f_vect2,f_vect3,f_vect4,dfdx_vect1,dfdx_vect2,dfdx_vect3,dfdx_vect4


    f1,f2,f3,f4,df1,df2,df3,df4=make_function(FNN)

    eq1=df1(params[:,0],inputs)-k[0]*f1(params[:,0],inputs)*f3(params[:,2],inputs)+(k[1]+k[2])*f4(params[:,3],inputs)*f1(params[:,0],inputs)+k[3]*f1(params[:,0],inputs)*f3(params[:,2],inputs)-k[4]*f1(params[:,0],inputs)*f2(params[:,1],inputs)-(k[5]+k[6])*f3(params[:,2],inputs)*f2(params[:,1],inputs)-k[8]*f3(params[:,2],inputs)-k[9]*f2(params[:,1],inputs)

    eq2=df2(params[:,1],inputs)-k[3]*f3(params[:,2],inputs)*f1(params[:,0],inputs)+k[4]*f2(params[:,1],inputs)*f1(params[:,0],inputs)+(k[5]+k[6])*f2(params[:,1],inputs)*f3(params[:,2],inputs)+k[7]*f2(params[:,1],inputs)*f4(params[:,3],inputs)+k[9]*f2(params[:,1],inputs)

    eq3=df3(params[:,2],inputs)+(k[0]+k[3])*f3(params[:,2],inputs)*f1(params[:,0],inputs)-(k[1]+k[2])*f4(params[:,3],inputs)*f1(params[:,0],inputs)-k[4]*f2(params[:,1],inputs)*f1(params[:,0],inputs)-(k[5]-k[6])*f2(params[:,1],inputs)*f3(params[:,2],inputs)-2.*k[7]*f2(params[:,1],inputs)*f4(params[:,3],inputs)+k[8]*f3(params[:,2],inputs)-k[9]*f2(params[:,1],inputs)

    eq4=df4(params[:,3],inputs)-k[0]*f3(params[:,2],inputs)*f1(params[:,0],inputs)+(k[1]+k[2])*f4(params[:,3],inputs)*f1(params[:,0],inputs)+k[7]*f2(params[:,1],inputs)*f4(params[:,3],inputs)-k[8]*f3(params[:,2],inputs)

    
    true_eq1=eq1**2.+IC[0]**2
    true_eq2=eq2**2.+IC[1]**2
    true_eq3=eq3**2.+IC[2]**2
    true_eq4=eq4**2.+IC[3]**2
    
    
    
    #return None
    #return sum((np.mean(abs(true_eq1**2.)),np.mean(abs(true_eq2**2.)),np.mean(abs(true_eq3**2.)),np.mean(abs(true_eq4**2))))
    return sum((np.mean(np.log10(abs(true_eq1)+1.)),np.mean(np.log10(abs(true_eq2)+1.)),np.mean(np.log10(abs(true_eq3)+1.)),np.mean(np.log10(abs(true_eq4)+1.))))
    

def get_coeff_ism_reduced(Tgas,flux):

    Te = Tgas*8.617343e-5 #Tgas in eV (eV)
    lnTe = np.log(Te) #ln of Te (#)
    invTe = 1.e0/Te #inverse of T (1/eV)

    small=1e-10

    k=slow_np.zeros(10)+small

    k[0] = small + (np.exp(-32.71396786e0+13.5365560e0*lnTe-5.73932875e0*(lnTe**2)+1.56315498e0*(lnTe**3)-0.28770560e0*(lnTe**4)+3.48255977e-2*(lnTe**5)-2.63197617-3*(lnTe**6)+1.11954395e-4*(lnTe**7)-2.03914985e-6*(lnTe**8)))

    if Tgas<=5.5e3:
        k[1] = small + (3.92e-13*invTe**0.6353e0)

    if Tgas>5.5e3:
        k[2]=small + (np.exp(-28.61303380689232e0-0.7241125657826851e0*lnTe-0.02026044731984691e0*lnTe**2-0.002380861877349834e0*lnTe**3-0.0003212605213188796e0*lnTe**4-0.00001421502914054107e0*lnTe**5+4.989108920299513e-6*lnTe**6+5.755614137575758e-7*lnTe**7-1.856767039775261e-8*lnTe**8-3.071135243196595e-9*lnTe**9))

    k[3] = small + (1.4e-18*Tgas**0.928*np.exp(-Tgas/16200.))

    k[4]=small + (np.exp(-18.01849334273e0+2.360852208681e0*lnTe-0.2827443061704e0*lnTe**2+0.01623316639567e0*lnTe**3-0.03365012031362999e0*lnTe**4+0.01178329782711e0*lnTe**5-0.001656194699504e0*lnTe**6+0.0001068275202678e0*lnTe**7-2.631285809207e-6*lnTe**8))
    
    if Tgas<=1.16e3:
        k[5]=small + (2.56e-9*Te**1.78186e0)

    if Tgas>1.16e3:
        k[6]=small + (np.exp(-20.37260896533324e0+1.139449335841631e0*lnTe-0.1421013521554148e0*lnTe**2+0.00846445538663e0*lnTe**3-0.0014327641212992e0*lnTe**4+0.0002012250284791e0*lnTe**5+0.0000866396324309e0*lnTe**6-0.00002585009680264e0*lnTe**7+2.4555011970392e-6*lnTe**8-8.06838246118e-8*lnTe**9))

    if Tgas>=1e1 and Tgas<=1e7:
        k[7]=small + ((2.96e-6/np.sqrt(Tgas)-1.73e-9+2.50e-10*np.sqrt(Tgas)-7.77e-13))

    k[8],k[9]= interp_photorate('NN_jax/HM_flux_rate.dat',flux)






    return k

def interp_photorate(filename,flux_input):

    flux,k1,k2=slow_np.loadtxt(filename,unpack=True)
    
    near_flux=np.min(abs(flux_input-flux))
    near_flux_index=slow_np.argmin(abs(flux_input-flux))
    if near_flux<flux_input:
        k1_new=(k1[near_flux_index]+k1[near_flux_index+1])/2.
        k2_new=(k2[near_flux_index]+k2[near_flux_index+1])/2.
    if near_flux==flux_input:
        k1_new=k1[near_flux_index]
        k2_new=k2[near_flux_index]
    if near_flux>flux_input:
        k1_new=(k1[near_flux_index]+k1[near_flux_index-1])/2.
        k2_new=(k2[near_flux_index]+k2[near_flux_index-1])/2.

    return k1_new,k2_new
