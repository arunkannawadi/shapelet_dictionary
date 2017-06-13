import numpy as np
import math
import matplotlib.pyplot as plt
from pylab import imshow, meshgrid
import matplotlib.cm as cm
from scipy import special

Pi = math.pi
factorial = math.factorial

def Coeff(N, M, beta):
    """ Determine the normalising coefficients that appear in front of the Laguerre polynomials.

        @param N        Energy quantum number
        @param M        Azimuthal quantum number

        @returns        Two coefficients, both of which need to multiply the Laguerre polynomials
    """

    B = (-1)**((N - np.abs(M))/2) / beta**(np.abs(M) + 1)
    C = (float(factorial(int((N-np.abs(M))/2)))/Pi/float(factorial((N+np.abs(M))/2)))**(.5)
    return B, C

def polar_shapelets_real(N, M, beta):

    coeff_1, coeff_2 = Coeff(N, M, beta)    
    gen_laguerre = special.genlaguerre(n = (N-np.abs(M))/2, alpha = np.abs(M))

    Laguer_N_M = lambda x, phi: coeff_1 * coeff_2 * x**(np.abs(M)) * gen_laguerre(x**2/beta**2) * np.exp(-x**2/2./beta**2) * np.exp(-1j*M*phi).real
    
    return Laguer_N_M

def polar_shapelets_imag(N, M, beta):
    
    coeff_1, coeff_2 = Coeff(N,M,beta)

    gen_laguerre = special.genlaguerre(n = (N-np.abs(M))/2, alpha = np.abs(M))

    Laguer_N_M = lambda x, phi: coeff_1 * coeff_2 * x**(np.abs(M)) * gen_laguerre(x**2/beta**2) * np.exp(-x**2/2./beta**2) * np.exp(-1j*M*phi).imag
    
    return Laguer_N_M

def plot_shapelets(N,M, beta):
    
    X = np.linspace(-5, 5, 100)
    Y = np.linspace(-5, 5, 100)
    Xv, Yv = meshgrid(X,Y)
    R = np.sqrt(Xv**2 +  Yv**2)
    Phi = np.zeros_like(R)
    for i in xrange(np.shape(Xv)[0]):
        for j in xrange(np.shape(Xv)[1]):
            Phi[i,j] = math.atan2(Yv[i,j], Xv[i,j])

    Func = polar_shapelets_imag(N, M, beta)(R, Phi)
    im = imshow(Func, cmap = cm.bwr)
    plt.show()
    
if __name__ == "__main__":
    
    
    plot_shapelets(6,6, 1)
    

    #print(polar_shapelets(1,1,0,Pi/2, 1.))     
   
