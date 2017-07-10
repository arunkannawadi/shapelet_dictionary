"""
------------------
berry == Berry et al. MNRAS 2004
refregier == Refregier MNRAS 2003 / Shapelets I and II
bosch == Bosch J. AJ 2010 vol 140 pp 870 - 879
------------------
"""

import sys,os
import numpy as np
from scipy.special import hermitenorm
from scipy.integrate import quad


from utils.shapelet_utils import *
from utils.galsim_utils import *

import pdb; pdb.set_trace()

## About the warning of the a lot of images
#import matplotlib
#matplotlib.use("Agg")

import pyfits
import galsim
import math


## Custom solver functions
from solver_routines import *
from plotting_routines import *

DEBUG = 0

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
    Xv,Yv = np.meshgrid(X,Y)
    for m in xrange(M):
      for n in xrange(M):
        arr = elliptical_shapelet(m,n,sx=1.,sy=1.,theta=1.*np.pi/4.)(Xv,Yv)
        ax[m,n].imshow(arr,cmap=cm.bwr,vmax=1.,vmin=-0.5)
        ax[m,n].set_title(str(m)+','+str(n))
    plt.savefig('test.png')

def shapelet_decomposition(image_data,\
        f_path = '/home/',\
        N1=20,N2=20, basis = 'XY', solver = 'omp', image = None, \
        coeff_0 = None, noise_scale = None, \
        alpha_ = None, Num_of_shapelets = 21, \
        select_img_idx = 91, n_max = None,\
        column_number = 1.01, plot_decomp= False, \
        beta_array = [1.5, 2, 2.5]):

    """ 
    Do the shapelet decomposition and return the reconstructed image and coefficint vector;
    If noise_scale is zero (initial image) then return the label_array, and beta_aray
    
    Parameters:
    ----------
    image_data : Array / mutable object in python, just to store centroid values and sigma
                      of the noiseless image for the future decomp.
    f_path : Path variable for saving the image decomp.
    N1,N2 : n and m quantum numbers respectively
    basis : do decomposition in this basis
        -- XY - Standard Descartes coordinate shapelet space
        -- Polar - Polar coordinate shapelet space
        -- XY_Elliptical - Elllipse in XY shapelet space
        -- Polar_Elliptical - Ellipse in Polar shapelets space
    solver : choose an algorithm for fitting the coefficients
        -- SVD - Singular Value Decomposition
        -- omp - using the Orthogonal Matching Pursuit
        -- P_2 - Standard least squares
        -- P_1 - Lasso regularization technique
    image : Image to be decomposed
    coeff_0 : Coefficients of the 0 noise decomposition
    noise_scale : A number which multiplies the noise_matrix
    alpha_ : Scalar factor in fron of the l_1 norm in the lasso_regularization method
    Num_of_shapelets : Number which refers to maximum allowed number for OMP method to use
    n_max : This nubmer refers to the maximum order of shapelets that could be used in decomp.
    column_number : Just used for making distinction for images noised by differend matrices
    plot_decomp : Should the plot_solution and plot_decomposition be used or not
    beta_array : Array of beta values to be used for compound basis
    
    To add optional keyword*:
    -------------------------
    q : Axis ratio for elliptical coordinates, defined as q = b / a
    theta : Direction angle of the ellipse

    """
    ## Flag for test stability
    ## controling the change of image_data nad label_arr and cube
    flag_test = False
    ## Obtaining galaxy images
    if np.all(image == None):
        ## Aquireing the noiseless image
        flag_test = True
        cube = pyfits.getdata('../../data/cube_real_noiseless.fits')
        background = 1.e6*0.16**2
        img = galsim.Image(78,78) # cube has 100, 78x78 images
        size_X = 78; size_Y = 78
        pick_an_img = [select_img_idx]
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
            if (os.path.isfile('Plots/Initial_image_stability.png')):
                pass
            else:
                from pylab import imshow
                imshow(cube[img_idx])
                plt.savefig('Plots/Initial_image_stability.png')
                plt.clf() 
        
        img = galsim.Image(cube[img_idx],scale = 1.0, xmin=0,ymin=0)
 
        ## Here catch an exception and exit the function if the FindAdaptiveMom doesn't converge
        try:
            shape = img.FindAdaptiveMom() #strict = False, watch out for failure, try block
        except RuntimeError as error:
            print("RuntimError: {0}".format(error))
            if noise_scale == 0:
                return [None]*5
            else:
                return [None]*2
        
        ## Remember this from the 0 noise image
        if flag_test:
            x0, y0, sigma, theta, q = get_moments(shape)
            image_data[0] = x0; image_data[1] = y0; image_data[2] = sigma
            image_data[3] = theta; image_data[4] = q
        else:
            x0 = image_data[0]
            y0 = image_data[1]
            sigma = image_data[2] 
            theta = image_data[3]
            q = image_data[4]
        
        ## In order for function calls to be consistent
        ## Make this array even if the basis is not 'Compound'
        ## To enable correct plotting
        if basis == 'Compound':
            beta_array = [sigma/4.,sigma/2., sigma, 2*sigma, 4*sigma]
        else:
            beta_array = [sigma]

        ## Initialize the basis matrix size and base_coefs sizes according
        ## to the basis used
        if (basis == 'XY') or (basis == 'XY_Elliptical'):
            D = np.zeros((size_X*size_Y,N1*N2)) # alloc for Dictionary
            base_coefs = np.zeros((N1,N2))
        elif (basis == 'Polar') or (basis == 'Polar_Elliptical'):
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
            n_max = Num_of_shapelets

        signal = cube[img_idx].flatten() 
        shapelet_reconst = np.zeros((len(beta_array), size_X*size_Y))
        residual = np.zeros((len(beta_array), size_X*size_Y))
        residual_energy_fraction = np.zeros(len(beta_array))
        recovered_energy_fraction = np.zeros(len(beta_array))
        ## -------------------------------------------------------------
        ## Labeling could also be done inside the plot_stability routine
        ## just take the index of coeff_stability:
        ## k --> n = k/N1, m = k%N1 for cartesian
        ## k --> n,m = from indices in the adapted pascal triangle for polar
        ## -------------------------------------------------------------
        label_arr = np.chararray(D.shape[1], itemsize=10)

        ## Decompose into Polar/Polar_Elliptical / XY /XY_Elliptical / Compound / w/ inner product
        if (basis == 'Polar') or (basis == 'Polar_Elliptical'):
            shapelet_reconst[0] = decompose_polar(basis,\
                    D,base_coefs,\
                    shapelet_reconst[0], signal, flag_test, \
                    label_arr,\
                    n_max, N1,N2,\
                    x0,y0,sigma,\
                    X,Y,\
                    q=q, theta = theta)
        elif (basis == 'XY') or (basis == 'XY_Elliptical'):
            shapelet_reconst[0] = decompose_cartesian(basis,\
                    D,base_coefs,\
                    shapelet_reconst[0], signal, flag_test, \
                    label_arr,\
                    n_max,N1,N2,\
                    x0,y0,sigma,\
                    X,Y,\
                    q=q,theta = theta)

        elif(basis == 'Compound'):
            
            shapelet_reconst = decompose_compound(basis,\
                    D,base_coefs,beta_array, \
                    shapelet_reconst, signal, flag_test, \
                    label_arr,\
                    n_max, N1,N2,\
                    x0,y0,sigma,\
                    X,Y,\
                    polar_basis = 'refregier',\
                    q=q, theta = theta)
        
        for i in xrange(len(beta_array)):
            residual[i]= signal - shapelet_reconst[i]
            residual_energy_fraction[i] = np.sum(residual[i]**2)/np.sum(signal**2)
            recovered_energy_fraction[i] = np.sum(shapelet_reconst[i]**2)/np.sum(signal**2)

        if basis =='Polar' or basis == 'Polar_Elliptical':
            print "Comparing moments_amp to base_coefs[0]: ", \
                    np.abs(base_coefs[0]), shape.moments_amp
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
            
            file_path_check = \
                    f_path + \
                    'Decomp/' + basis +'_'\
                    +str(n_max) + '_' + str(column_number) +'_.png'
            
            ## Check if there is already the XY decomp
            if (os.path.isfile(file_path_check) == False) and flag_test:
                
                ## Plot the inner product decomposition only once

                plot_decomp = False
                plot_decomposition(basis, cube, img_idx, size_X, size_Y, \
                        base_coefs, N1, N2,\
                        shapelet_reconst, signal, \
                        residual, residual_energy_fraction ,recovered_energy_fraction, \
                        f_path + 'Decomp/_' + basis +'_'\
                        +str(n_max) + '_' + str(column_number),\
                        beta_array = beta_array)

        reconst, coeffs = select_solver_do_fitting_plot(\
                f_path, basis, coeff_0, noise_scale, \
                N1,N2,n_max,column_number,\
                cube[img_idx],D,signal,solver, beta_array,\
                Num_of_shapelets = Num_of_shapelets, alpha_ = alpha_, plot = True)

        if noise_scale == 0:
            return cube[img_idx], reconst, coeffs, label_arr, beta_array
        else:
            return reconst, coeffs

if __name__ == "__main__":   
    
    show_some_shapelets()
    #p_shapelet.plot_shapelets(6,2,1)
    #check_orthonormality()
