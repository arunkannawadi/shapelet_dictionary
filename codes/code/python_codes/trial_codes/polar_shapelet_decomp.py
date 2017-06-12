import numpy as np
import math
import matplotlib.pyplot as plt
from pylab import imshow, meshgrid
import matplotlib.cm as cm
from scipy import special


#-----------------
# berry == Berry et al. MNRAS 2004
# refregier == Massey & Refregier MNRAS 2005
#----------------

Pi = math.pi
factorial = math.factorial

def coeff(N, M, beta):
    B = (-1)**((N - np.abs(M))/2) / beta**(np.abs(M) + 1)
    C = (float(factorial(int((N-np.abs(M))/2)))/(Pi*float(factorial((N+np.abs(M))/2))))**(.5)
    return B, C

def polar_shapelets_refregier(N, M, beta):

    coeff_1, coeff_2 = coeff(N, M, beta)
    gen_laguerre = special.genlaguerre(n=(N-np.abs(M))/2, alpha=np.abs(M))

    laguer_N_M = lambda x, phi: coeff_1 * coeff_2 * x**(np.abs(M)) * gen_laguerre(x**2/beta**2) * np.exp(-x**2/2./beta**2) * np.exp(-1j*M*phi)
    
    return laguer_N_M


def polar_shapelets_real(N, M, beta):

    coeff_1, coeff_2 = coeff(N, M, beta)
    gen_laguerre = special.genlaguerre(n=(N-np.abs(M))/2, alpha=np.abs(M))

    laguer_N_M = lambda x, phi: coeff_1 * coeff_2 * x**(np.abs(M)) * gen_laguerre(x**2/beta**2) * np.exp(-x**2/2./beta**2) * np.exp(-1j*M*phi).real
    
    return laguer_N_M

def polar_shapelets_imag(N, M, beta):
    
    coeff_1, coeff_2 = coeff(N,M,beta)

    gen_laguerre = special.genlaguerre(n = (N-np.abs(M))/2, alpha = np.abs(M))

    laguer_N_M = lambda x, phi: coeff_1 * coeff_2 * x**(np.abs(M)) * gen_laguerre(x**2/beta**2) * np.exp(-x**2/2./beta**2) * np.exp(-1j*M*phi).imag
    
    return laguer_N_M

#From Berry et al. Eq (12)

def coeff_berry(N,M,beta):
    return (2.*float(factorial((N-np.abs(M)/2)))/(Pi*beta**2*float(factorial((N+np.abs(M))/2))))**(0.5)

def polar_shapelets_berry(N,M, beta):

    coeff = coeff_berry(N,M,beta)
    gen_laguerre = special.genlaguerre(n = (N-np.abs(M))/2, alpha = np.abs(M))

    laguer_N_M = lambda r, phi: coeff * (r**2/beta**2)**(np.abs(M)/2) * np.exp(-(r**2/beta**2)/2.) * gen_laguerre(r**2/beta**2) * np.exp(-1j*M*phi)
    
    return laguer_N_M


def plot_shapelets(N,M, beta):
    
    X = np.linspace(-5, 5, 100)
    Y = np.linspace(-5, 5, 100)
    Xv, Yv = meshgrid(X,Y)
    R = np.sqrt(Xv**2 +  Yv**2)
    phi = np.zeros_like(R)
    for i in xrange(np.shape(Xv)[0]):
        for j in xrange(np.shape(Xv)[1]):
            phi[i,j] = math.atan2(Yv[i,j], Xv[i,j])

    func = polar_shapelets_berry(N, M, beta)(R, phi).real
    im = imshow(func, cmap=cm.bwr)
    plt.show()

def check_orthonormality():
    
    X = np.linspace(-5, 5, 100)
    Y = np.linspace(-5, 5, 100)
    Xv, Yv = meshgrid(X,Y)
    R = np.sqrt(Xv**2 +  Yv**2)
    phi = np.zeros_like(R)
    for i in xrange(np.shape(Xv)[0]):
        for j in xrange(np.shape(Xv)[1]):
            phi[i,j] = math.atan2(Yv[i,j], Xv[i,j])
    
    print(\
            np.abs(\
            np.dot(\
            polar_shapelets_refregier(2,2,1)(R,phi).flatten(),\
            polar_shapelets_refregier(2,2,1)(R,phi).flatten())), \
            np.abs(\
            np.dot(\
            polar_shapelets_berry(2,2,1)(R,phi).flatten(), \
            polar_shapelets_berry(2,2,1)(R,phi).flatten()) ))

if __name__ == "__main__":
    
  
    plot_shapelets(6,6, 1) 
    check_orthonormality() 
    #print(polar_shapelets(1,1,0,Pi/2, 1.))     
   
