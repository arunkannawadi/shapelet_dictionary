import sys,os
import numpy as np
from scipy.special import hermitenorm
from scipy.integrate import quad
from utils.shapelet_utils import *

## About the warning of the a lot of images
import matplotlib
matplotlib.use("Agg")

import pyfits
import galsim
import math

import pdb; pdb.set_trace()

##------------------
## berry == Berry et al. MNRAS 2004
## refregier == Refregier MNRAS 2003 / Shapelets I and II
##------------------


import trial_codes.polar_shapelet_decomp as p_shapelet


## Custom solver functions
from solver_routines import *
from plotting_routines import *

DEBUG = 0

##Define orthonormal basis - shapelets
def shapelet1d(n,x0=0,s=1):

    """Make a 1D shapelet template to be used in the construction of 2D shapelets

    @param n Energy quantum number
    @param x0 centroid
    @param s same as beta parameter in refregier

    """
    def sqfac(k):
        fac = 1.
        for i in xrange(k):
            fac *= np.sqrt(i+1)
        return fac

    u = lambda x: (x-x0)/s
    fn = lambda x: (1./(2*np.pi)**0.25)*(1./sqfac(n))*hermitenorm(n)(u(x))*np.exp(-0.25*u(x)**2) 
    return fn

def shapelet2d(m,n,x0=0,y0=0,sx=1,sy=1):
    
    """Make a 2D shapelet template function to be used in image decomposition later on
    
    @param n Energy quatum number
    @param m Magnetic quantum number
    @param x0 image centroid - X coordinate
    @param y0 image centroid - Y coordinate
    @param sx beta scale for the X shapelet space
    @param xy beta scale for the Y shapelet space 
    
    """

    u = lambda x: (x-x0)/sx
    v = lambda y: (y-y0)/sy
    fn = lambda x,y: np.outer(shapelet1d(m)(u(x)),shapelet1d(n)(v(y)))
    return fn

def elliptical_shapelet(m,n,x0=0,y0=0,sx=1,sy=1,theta=0):
    
    """ Make elliptical shapelets to be used in the decomposition of images later on

    @param n Energy quantum number
    @param m Magnetic quantum number
    @param x0 image centroid - X coordinate
    @param y0 image centroid - Y coordinate
    @param sx beta scale for X shapelet space (?)
    @param sy beta scale for Y shapelet space (?)
    @param theta true anomaly
    
    """
    u = lambda x,y: (x-x0)*np.cos(theta)/sx + (y-y0)*np.sin(theta)/sy
    v = lambda x,y: (y-y0)*np.cos(theta)/sy - (x-x0)*np.sin(theta)/sx
    fn = lambda x,y: np.outer(shapelet1d(m)(u(x,y)),shapelet1d(n)(v(x,y)))
    return fn

def check_orthonormality():
    M,N = 3, 3
    X = np.arange(-16,16,0.01)
    Y = np.arange(-16,16,0.01)

    for k1 in xrange(M*N):
        m1,n1 = k1/N, k1%N
        b1 = shapelet2d(m1,n1,x0=0.3,y0=0.4,sx=2,sy=3)(X,Y)
        for k2 in xrange(M*N):
            m2,n2 = k2/N, k1%N
            b2 = shapelet2d(m2,n2,x0=0.3,y0=0.4,sx=2,sy=3)(X,Y)

            print m1-m2,n1-n2,np.sum(b1*b2)*(0.01/2)*(0.01/3)

def calculate_spark(D):
    pass

def show_some_shapelets(M=4,N=4):
    fig,ax = plt.subplots(M,N)
    X = np.linspace(-8,8,17)
    Y = np.linspace(-8,8,17)
    for m in xrange(M):
      for n in xrange(M):
        arr = shapelet2d(m,n)(X,Y)
        ax[m,n].imshow(arr,cmap=cm.bwr,vmax=1.,vmin=-0.5)
        ax[m,n].set_title(str(m)+','+str(n))
    plt.show()

def decompose_cartesian(basis,\
        D, base_coefs, \
        shapelet_reconst, signal, noise_scale, \
        label_arr,\
        n_max,N1,N2,\
        x0,y0,sigma,\
        X,Y,\
        a=1., b=1.):
    """
    Decompose into XY or Elliptical basis, return obtained reconstruction with shapelets
    and also construct label_array and basis matrix D
    """
    max_order = get_max_order(basis,n_max)
    for k in xrange(N1*N2):
        m,n = k/N1, k%N1
        if (m+n <= max_order): 
            if noise_scale == 0:
                ## To be consistent with indexation of the basis matrix
                ## D
                label_arr[k] = (str("(%d, %d)" % (n,m)))
            if basis == 'XY':
                arr = shapelet2d(m,n,x0=x0,y0=y0,sx=sigma,sy=sigma)(X,Y).flatten() 
            elif basis == 'Elliptical':
                arr = elliptical_shapelet(m,n,x0,y0, sx=a, sy=b)(X,Y).flatten()
            D[:, k] = arr
            arr_norm2 = np.dot(arr,arr)
            coef = np.dot(arr,signal)
            if(coef==0):
                base_coefs[n,m] =0
            else:
                base_coefs[n,m] = coef/np.sqrt(arr_norm2)
                shapelet_reconst = shapelet_reconst + (coef*arr)/arr_norm2

    return shapelet_reconst

def decompose_polar(basis,\
       D,base_coefs, \
       shapelet_reconst, signal, noise_scale, \
       label_arr,\
       n_max, N1,N2,\
       x0,y0,sigma,\
       R,Phi):

    """
    Decompose into polar shapelet basis, return shapelet reconstruction
    and construct basis matrix as well as label_array
    """
    ## Set the counter for columns of basis matrix D

    polar_basis = 'refregier' 
    k_p = 0

    ## For symmetric coeffs triangle solve
    ## n^2 + 2n + 2 = 2n_max, positive solution, such that
    ## Sum(i+1)_{i in [0,n]} <= n_max
    max_order = get_max_order(basis,n_max)
    
    ## This needs to go to max_order + 1, because maximum order would not be included
    ## in the iteration
    for n in xrange(max_order + 1):
        for m in xrange(-n,n+1,2):
            
            ## n_max - defined as:
            ## theta_max (galaxy size) / theta_min (smallest variation size) - 1  

            if noise_scale==0:
                ## To be consistent with the indexation of the basis matrix
                ## D
                label_arr[k_p] = (str("(%d, %d)" % (n,m)))
            
            if (polar_basis == 'refregier'):
                arr_res = \
                        p_shapelet.polar_shapelets_refregier(n,m,sigma)(R,Phi).flatten()    
            elif (polar_basis == 'berry'):
                arr_res = \
                        p_shapelet.polar_shapelets_berry(n,m,sigma)(R,Phi).flatten()
            ## Make the basis matrix D
            D[:,k_p] = arr_res 
            ## Calculate the norms of basis vectors and coefficients
            arr_norm2_res = np.dot(arr_res,arr_res)
            coef_res= np.dot(arr_res, signal)

            ## Add coefficients to basis_coefs array for later use
            ## Make the shapelet reconstruction
            if (coef_res==0): 
                base_coefs[k_p] = 0 
            else: 
                base_coefs[k_p] = coef_res/np.sqrt(arr_norm2_res)
                shapelet_reconst = shapelet_reconst \
                        + coef_res*arr_res/arr_norm2_res
            k_p += 1
    
    return shapelet_reconst

def shapelet_decomposition(image_data,\
        f_path = '/home/',\
        N1=20,N2=20, basis = 'XY', solver = 'sparse', image = None, \
        coeff_0 = None, noise_scale = None, \
        alpha_ = None, Num_of_shapelets = None, n_max = None,\
        column_number = 1.01, plot_decomp= False, \
        q = 2., beta_array = [1.5, 2, 2.5]):

    """ 
    Do the shapelet decomposition
    
    @param image_data Array / mutable object in python, just to store centroid values and sigma
                      of the noiseless image for the future decomp.
    @param f_path Path variable for saving the image decomp.
    @param N1,N2 n and m quantum numbers respectively
    @param basis do decomposition in this basis
        -- XY - Standard Descartes coordinate shapelet space
        -- Polar - Polar coordinate shapelet space
        -- Elliptical - Elliptical coordinate shapelet space
    @param solver choose an algorithm for fitting the coefficients
        -- SVD - Singular Value Decomposition
        -- sparse - using the Orthogonal Matching Pursuit
        -- P_2 - Standard least squares
        -- P_1 - Lasso regularization technique
    @param image Image to be decomposed
    @param coeff_0 Coefficients of the 0 noise decomposition
    @param noise_scale A number which multiplies the noise_matrix
    @param alpha_ Scalar factor in fron of the l_1 norm in the lasso_regularization method
    @param Num_of_shapelets Number which refers to maximum allowed number for OMP method to use
    @param n_max This nubmer refers to the maximum order of shapelets that could be used in decomp.
    @param column_number Just used for making distinction for images noised by differend matrices
    @param plot_decomp Should the plot_solution and plot_decomposition be used or not
    @param q Axis ratio for elliptical coordinates, defined as q = b / a
    @param theta Direction angle of the ellipse

    @returns reconstructed image and coefficients of the selected method along with labels of shapelets
    """

    ## Obtaining galaxy images
    if (image == None) or (noise_scale == 0):
        #cube = pyfits.getdata('../../data/cube_real.fits')
        ## Aquireing the noiseless image
        cube = pyfits.getdata('../../data/cube_real_noiseless.fits')
        background = 1.e6*0.16**2
        img = galsim.Image(78,78) # cube has 100, 78x78 images
        size_X = 78; size_Y = 78
        pick_an_img = [91]
    else:
        ## Added this part for stability test with noise
        ## Background is zero because I already substracted background in the first decomp
        size_X = np.shape(image)[0]; size_Y = np.shape(image)[1]
        cube = [image]
        background = 0
        pick_an_img = [0] 

    X = np.linspace(0,size_X-1,size_X)  
    Y = np.linspace(0,size_Y-1,size_Y)

    for img_idx in pick_an_img:

        ## Take the image, reduce for background, find moments set n_max

        cube[img_idx] -= background
        
        ## Just for checking plot the chosen image
        if noise_scale == 0:
            if os.path.isfile('Plots/Initial_image.png'):
                pass
            else:
                from pylab import imshow
                imshow(cube[img_idx])
                plt.savefig('Plots/Initial_image.png')
                plt.clf()
        
        img = galsim.Image(cube[img_idx],xmin=0,ymin=0)
        
        ## Here catch an exception and exit the function if the FindAdaptiveMom doesn't converge
        try:
            shape = img.FindAdaptiveMom() #strict = False, watch out for failure, try block
        except RuntimeError as error:
            print("RuntimError: {0}".format(error))
            return None, None,None
        
        ## Remember this from the 0 noise image
        if noise_scale == 0:
            x0,y0 = shape.moments_centroid.x, shape.moments_centroid.y ## possible swap b/w x,y
            sigma = shape.moments_sigma
            image_data[0] = x0; image_data[1] = y0; image_data[2] = sigma
        else:
            x0 = image_data[0]
            y0 = image_data[1]
            sigma = image_data[2]
        
        ## In order for function calls to be consistent
        ## Make this array even if the basis is not 'Compound'
        ## To enable correct plotting
        if basis == 'Compound':
            beta_array = [sigma/4.,sigma/2., sigma, 2*sigma, 4*sigma]
        else:
            beta_array = [sigma]


        ## Initialize the basis matrix size and base_coefs sizes according
        ## to the basis used
        if (basis == 'XY') or (basis == 'Elliptical'):
            D = np.zeros((size_X*size_Y,N1*N2)) # alloc for Dictionary
            base_coefs = np.zeros((N1,N2))
        elif (basis == 'Polar'):
            D = np.zeros((size_X*size_Y,N1*N2))#, dtype=complex)
            base_coefs = np.zeros(N1*N2)#, dtype=complex)
        elif (basis == 'Compound'):
            ## D must be this size because different betas are included
            D = np.zeros((size_X*size_Y, len(beta_array)*N1*N2))
            ## base_coefs must be this size for the representation
            base_coefs = np.zeros((N1,N2*len(beta_array)))


        if n_max == None:
            ## So that it is a complete triangle
            ## Formula for a symmetric triangle is n * (n+2)
            n_max = 21

        ## Make a meshgrid for the polar shapelets

        Xv, Yv = np.meshgrid((X-x0),(Y-y0))
        R = np.sqrt(Xv**2 + Yv**2)
        
        Phi = np.zeros_like(R)
        for i in xrange(np.shape(Xv)[0]):
            for j in xrange(np.shape(Xv)[1]):
                Phi[i,j] = math.atan2(Yv[i,j], Xv[i,j])

        signal = cube[img_idx].flatten() 
        shapelet_reconst = np.zeros((len(beta_array), size_X*size_Y))
        residual = np.zeros((len(beta_array), size_X*size_Y))
        residual_energy_fraction = np.zeros(len(beta_array))
        recovered_energy_fraction = np.zeros(len(beta_array))
        ## -------------------------------------------------------------
        ## Labeling could also be done inside the plot_stability routine
        ## just take the index of coeff_stability:
        ## k --> n = k/N1, m = k%N1 for cartesian
        ## k --> n,m = from indices in the pascal triangle
        ## -------------------------------------------------------------
        label_arr = np.chararray(D.shape[1], itemsize=10)

        ## Decompose into Polar / XY /Elliptical / Compound / w/ inner product
        if (basis == 'Polar'):
            shapelet_reconst[0] = decompose_polar(basis,\
                    D,base_coefs,\
                    shapelet_reconst[0], signal, noise_scale, \
                    label_arr,\
                    n_max, N1,N2,\
                    x0,y0,sigma,\
                    R,Phi)
        elif (basis == 'XY') or (basis == 'Elliptical'):
             
            a = sigma / np.sqrt(q)
            b = sigma * np.sqrt(q)
            shapelet_reconst[0] = decompose_cartesian(basis,\
                    D,base_coefs,\
                    shapelet_reconst[0], signal, noise_scale, \
                    label_arr,\
                    n_max,N1,N2,\
                    x0,y0,sigma,\
                    X,Y, a=a, b=b)

        elif(basis == 'Compound'):
            ## Just put different betas (range TBD) basis into a big basis matrix 
            ## let the algorithms pick the weights. Watch out that the order stays symmetric 
            step = 0
            max_order = get_max_order(basis, n_max)
            for sigma in beta_array:
                ## Reset indexation for a new basis
                ## with step the basis block is controled
                ##      |vec_basis_1_1   ... / vec_basis_2_1   .. /     |
                ##      |vec_basis_1_2   ... / vec_basis_2_2   .. /     |
                ## D=   .   .    ...         .  ..             .. .  ...|
                ##      |vec_basis_1_N-1 ... / vec_basis_2_N-1 .. /     |
                ##      |vec_basis_1_N   ... / vec_basis_2_N   .. /     |
                ## 
                ## vec_basis_i_j is actually arr variable which is the basis
                ## vector of ith (i is in len(beta_array), 
                ## corresponds to the number of beta values) basis 
                ## and j is it's jth coordinate value
                a = sigma / np.sqrt(q)
                b = sigma * np.sqrt(q)
                for k in xrange(N1*N2):
                    ## Number of cols of D is N1*N2*len(beta_array)
                    ## beta should change when one whole basis is sweeped by k
                    m,n = k/N1, k%N1
                    if (m+n <= max_order): 
                        if noise_scale == 0:
                            label_arr[k+(N1*N2)*step] = (str("(%d, %d)" % (n,m)))
                        arr = elliptical_shapelet(m,n,x0,y0,sx=a,sy=b)(X,Y).flatten()
                        D[:,k+(N1*N2)*step] = arr
                        arr_norm2 = np.dot(arr,arr)
                        coef = np.dot(arr, signal)
                        if coef == 0:
                            ## Watch not to overwrite the existing 
                            base_coefs[n, m + N2*step] = 0
                        else:
                            base_coefs[n, m + N2*step] = coef / np.sqrt(arr_norm2)
                            shapelet_reconst[step] += coef*arr/arr_norm2
                ## Basis finished increase the step
                step += 1
            
            pass

        for i in xrange(len(beta_array)):
            residual[i]= signal - shapelet_reconst[i]
            residual_energy_fraction[i] = np.sum(residual[i]**2)/np.sum(signal**2)
            recovered_energy_fraction[i] = np.sum(shapelet_reconst[i]**2)/np.sum(signal**2)

        if basis =='Polar':
            print "Comparing moments_amp to lowest order base_coefs", \
                    np.abs(base_coefs[np.where(base_coefs!=0)[0][0]] ), shape.moments_amp
            print "Base coefficients sum over signal", \
                (np.sum(base_coefs**2))/(np.sum(signal**2)), \
                (np.sum(residual**2)/np.sum(signal**2)) 
        else:
            print "Comparing moments_amp to base_coefs[0,0]", \
                    np.abs(base_coefs[0,0]), shape.moments_amp
            print "Base coefficients sum over signal", \
                (np.sum(base_coefs**2))/(np.sum(signal**2)), \
                (np.sum(residual**2)/np.sum(signal**2)) 

        ## Make the strings for nice representation in the output
        noise_scale_str = str("%.3e" % (noise_scale))
        if (alpha_ != None):
            alpha_str = str("%.3e" % (alpha_))

        mkdir_p(f_path + 'Decomp/')
        if (plot_decomp == True):
            
            ## Check if there is already the XY decomp
            if os.path.isfile(f_path + \
                    'Decomp/'+ solver+ '_' + basis +'_'\
                    +str(n_max) + '_' + str(column_number) +'_.png') == False:

                plot_decomposition(basis, cube, img_idx, size_X, size_Y, \
                        base_coefs, N1, N2,\
                        shapelet_reconst, signal, \
                        residual, residual_energy_fraction ,recovered_energy_fraction, \
                        f_path + 'Decomp/'+ solver+ '_' + basis +'_'\
                        +str(n_max) + '_' + str(column_number),\
                        beta_array = beta_array)

        reconst, coeffs = select_solver_do_fitting_plot(\
                f_path, basis, coeff_0, noise_scale, \
                N1,N2,n_max,column_number,\
                cube[img_idx],D,signal,solver, beta_array,\
                Num_of_shapelets = Num_of_shapelets, alpha_ = alpha_, plot_decomp = plot_decomp)

        if noise_scale == 0:
            return cube[img_idx], reconst, coeffs, label_arr, beta_array
        else:
            return reconst, coeffs

if __name__ == "__main__":   
    
    show_some_shapelets()
    #p_shapelet.plot_shapelets(6,2,1)
    #check_orthonormality()
