import numpy as np
import math
import matplotlib.pyplot as plt
from pylab import imshow, meshgrid
import matplotlib.cm as cm
from scipy import special
from scipy.integrate import quad,dblquad

#-----------------
# berry == Berry et al. MNRAS 2004
# refregier == Massey & Refregier MNRAS 2005
#----------------

Pi = math.pi
factorial = math.factorial

def coeff(N, M, beta):
    """ 
    Return normalization coefficients for refregier shapelets

    """
    A = (-1)**((N - np.abs(M))/2) / beta**(np.abs(M) + 1)
    B = (2*float(factorial(int((N-np.abs(M))/2)))/(float(factorial((N+np.abs(M))/2))))**(.5)
    C = .5*np.sqrt(np.abs(M)/Pi)
    return A,B, C

def polar_shapelets_refregier(N, M, beta):

    """
    Return callable function for normalized generalized Laguerre polynomials with separated 
    exp(-1j*M*phi) to Cos (M>0) and Sin (M<0) and ordinary radial part for M == 0
    """
    coeff_1, coeff_2, coeff_3 = coeff(N, M, beta)
    gen_laguerre = special.genlaguerre(n=(N-np.abs(M))/2, alpha=np.abs(M))
    
    if (M > 0):
        laguer_N_M = lambda x, phi: coeff_1 * coeff_2 * x**(np.abs(M)) * gen_laguerre(x**2/beta**2) * np.exp(-x**2/2./beta**2) * coeff_3 * np.cos(M*phi) 
    elif (M < 0):
        laguer_N_M = lambda x, phi: coeff_1 * coeff_2 * x**(np.abs(M)) * gen_laguerre(x**2/beta**2) * np.exp(-x**2/2./beta**2) * coeff_3 * np.sin(M*phi)
    elif (M == 0):
        laguer_N_M = lambda x, phi: coeff_1 * coeff_2 * x**(np.abs(M)) * gen_laguerre(x**2/beta**2) * np.exp(-x**2/2./beta**2)
    return laguer_N_M



"""
Trial function for testing
BEGIN
"""
def polar_shapelets_real(N, M, beta):

    coeff_1, coeff_2, coeff_3 = coeff(N, M, beta)
    gen_laguerre = special.genlaguerre(n=(N-np.abs(M))/2, alpha=np.abs(M))

    laguer_N_M = lambda x: coeff_1 *coeff_2 * x**(np.abs(M)) * gen_laguerre(x**2/beta**2) * np.exp(-x**2/(2*beta**2))
    return laguer_N_M

def polar_shapelets_imag(N, M, beta):
    
    coeff_1, coeff_2,coeff_3 = coeff(N,M,beta)

    laguer_N_M = lambda x: 0

    if (M>0):
        laguer_N_M = lambda x: coeff_3 * np.cos(M*x)
    elif (M<0):
        laguer_N_M = lambda x: coeff_3 * np.sin(M*x)
        
    return laguer_N_M

"""
Trial functions for testing
END
"""

#From Berry et al. Eq (12)

def coeff_berry(N,M,beta):
    return (2.*float(factorial((N-np.abs(M)/2)))/(beta**2*float(factorial((N+np.abs(M))/2))))**(0.5)

def polar_shapelets_berry(N,M, beta):

    coeff = coeff_berry(N,M,beta)
    gen_laguerre = special.genlaguerre(n = (N-np.abs(M))/2, alpha = np.abs(M))

    laguer_N_M = lambda r, phi: coeff * (r/beta)**(np.abs(M)) * np.exp(-(r**2/beta**2)/2.) * gen_laguerre(r**2/beta**2) * np.exp(-1j*M*phi)
    
    return laguer_N_M


def plot_shapelets(N,M, beta):

    """
    Make a 2D grid for visualizing the shapelets
    """
    X = np.linspace(-5, 5, 100)
    Y = np.linspace(-5, 5, 100)
    Xv, Yv = meshgrid(X,Y)
    R = np.sqrt(Xv**2 +  Yv**2)
    phi = np.zeros_like(R)
    for i in xrange(np.shape(Xv)[0]):
        for j in xrange(np.shape(Xv)[1]):
            phi[i,j] = math.atan2(Yv[i,j], Xv[i,j])

    func = polar_shapelets_refregier(N, M, beta)(R, phi)
    im = imshow(func, cmap=cm.bwr)
    plt.show()

if __name__ == "__main__":
    
  
    check = 0
    plot_shapelets(4,2, 1)
    

    if (check == 1):
        for n in xrange(10):
            for m in xrange(-n,n+1,2):
                print(n,m)
                #print('berry')
                #print(dblquad(lambda r,phi: r*polar_shapelets_berry(n,m,beta)(r,phi).conjugate()*polar_shapelets_berry(n,m,beta)(r,phi), 0, np.inf, lambda r: 0, lambda r: 2*Pi))
                print('refregier')
                print('radial_part')
                print(quad(lambda r: r*polar_shapelets_real(n,m,beta)(r)**2, 0, np.inf))
                print('angular_part')
                print(quad(lambda phi: polar_shapelets_imag(n,m,beta)(phi)**2,0, 2*Pi))
    #print(polar_shapelets(1,1,0,Pi/2, 1.))     
   
