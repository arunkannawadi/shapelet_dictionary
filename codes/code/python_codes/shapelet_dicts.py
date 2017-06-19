import sys
import numpy as np
import numpy.linalg as linalg
from scipy.special import hermitenorm
from scipy.integrate import quad
from sklearn import linear_model
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pyfits
import galsim
import math

import pdb; pdb.set_trace()

#------------------
# berry == Berry et al. MNRAS 2004
# refregier == Refregier MNRAS 2003 / Shapelets I and II
#------------------

from sklearn.linear_model import OrthogonalMatchingPursuit as OMP
import trial_codes.polar_shapelet_decomp as p_shapelet

# Set up global LaTeX parsing
plt.rc('text', usetex=True)
plt.rc('font', **{'family' : "sans-serif"})
params = {'text.latex.preamble' : [r'\usepackage{siunitx}', \
                r'\usepackage[utf8]{inputenc}', r'\usepackage{amsmath}']}
plt.rcParams.update(params)

DEBUG = 0

def mkdir_p(mypath):
    """
    Creates a directory. equivalent to using mkdir -p on the command line
    """

    from errno import EEXIST
    from os import makedirs,path
    import os

    root_path = os.path.dirname(os.path.realpath(__file__)) + '/'
    
    try:
        makedirs(root_path + mypath)
    except OSError as exc:
        if exc.errno == EEXIST and path.isdir(root_path + mypath):
            pass
        else: raise


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
    
def coeff_plot2d(coeffs,N1,N2,ax=None,fig=None,orientation='vertical'):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import LogNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if ax is None:
      fig, ax = plt.subplots()

    ### INPUT VALIDATION NEEDS TO BE FILLED IN

    coeffs_reshaped = np.abs(coeffs.reshape(N1,N2)) ## does not 
    coeffs_reshaped /= coeffs_reshaped.max()
    im = ax.imshow(coeffs_reshaped,cmap=cm.bwr,interpolation='none')
    
    # Force colorbars besides the axes
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size='5%')

    fig.colorbar(im,cax=cax,orientation=orientation)
    return fig,ax

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

def plot_decomposition(cube, img_idx, size_X, size_Y, \
        base_coefs,N1,N2,shapelet_reconst, signal, residual,\
        residual_energy_fraction ,recovered_energy_fraction, Path):

    """ Plot the decomposition obtained with the chosen __solver__

    @param cube array of images obtained from the .fits file
    @param img_idx image index in the array
    @param base_coefs base coefficients obtained from the decomposition
    @param N1,N2 number of coefficients used for n and m numbers respectively
    @param shapelet_reconst reconstruction of the image with the obtained base_coefs
    @param signal an image vector, obtained from flattening the original image matrix
    @param residual residual obtained with difference between signal and shapelet_reconst
    @param residual_energy_fraction energy fraction of the residual image
    @param recovered_energy_fraction energy fraction of the obtained image with shapelet_reconst
    @param basis variable which controls the selected __basis__ in which decomposition was made

    """ 
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, ax = plt.subplots(2,2, figsize = (10, 10))
    coeff_plot2d(base_coefs,N1,N2,ax=ax[1,1],fig=fig)
    vmin, vmax = min(shapelet_reconst.min(),signal.min()), max(shapelet_reconst.max(),signal.max())

    im00 = ax[0,0].imshow(cube[img_idx],vmin=vmin,vmax=vmax)
    im01 = ax[0,1].imshow(shapelet_reconst.reshape(size_X,size_Y),vmin=vmin,vmax=vmax)
    im10 = ax[1,0].imshow(residual.reshape(size_X,size_Y))

    # Force the colorbar to be the same size as the axes
    divider00 = make_axes_locatable(ax[0,0])
    cax00 = divider00.append_axes("right", size="5%")

    divider01 = make_axes_locatable(ax[0,1])
    cax01 = divider01.append_axes("right", size="5%")

    divider10 = make_axes_locatable(ax[1,0])
    cax10 = divider10.append_axes("right", size="5%")

    fig.colorbar(im00,cax=cax00); fig.colorbar(im01,cax=cax01); fig.colorbar(im10,cax=cax10)
    ax[0,0].set_title('Original (noisy) image')
    ax[0,1].set_title('Reconstructed image - Frac. of energy = '\
            +str(np.round(recovered_energy_fraction,4)))
    ax[1,0].set_title('Residual image - Frac. of energy = '\
            +str(np.round(residual_energy_fraction,4)))
    ax[1,1].set_title('Rel. magnitude of coefficients')
    fig.suptitle('Shapelet Basis decomposition')
    
    fig.tight_layout()
    plt.savefig(Path, dpi=200)
    plt.clf()

def plot_solution(N1,N2,cube, img_idx,size_X, size_Y,\
        reconst, residual, coefs,\
        recovered_energy_fraction, residual_energy_fraction, \
        n_nonzero_coefs, noise_scale, Path):

    """ Plot obtained images from the coefficients obtained with the selected __solver__
    
    @param cube array of images obtained from the .fits file
    @param img_idx image index in the array
    @param base_coefs base coefficients obtained from the decomposition
    @param N1,N2 number of coefficients used for n and m numbers respectively
    @param shapelet_reconst reconstruction of the image with the obtained base_coefs
    @param signal an image vector, obtained from flattening the original image matrix
    @param residual residual obtained with difference between signal and shapelet_reconst
    @param residual_energy_fraction energy fraction of the residual image
    @param recovered_energy_fraction energy fraction of the obtained image with shapelet_reconst
    @param basis variable which controls the selected __basis__ in which decomposition was made
    @param n_nonzero_coefs nonzero coefficients in the coefs variable
    @param fig figure object forwarded from plot_decomposition
    @param Path to control the __savefig__ path

    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig2, ax2 = plt.subplots(2,2, figsize = (10,10))
    vmin, vmax = min(reconst.min(),cube[img_idx].min()), max(reconst.max(),cube[img_idx].max())
    
    im00 = ax2[0,0].imshow(cube[img_idx], aspect = '1', vmin=vmin, vmax=vmax)
    im01 = ax2[0,1].imshow(reconst.reshape(size_X,size_Y), aspect = '1', vmin=vmin, vmax=vmax)
    im10 = ax2[1,0].imshow(residual.reshape(size_X,size_Y), aspect = '1')
    print coefs.shape
    coefs = coefs.reshape(2*N1,2*N2)
    coeff_plot2d(coefs,N1*2,N2*2,ax=ax2[1,1],fig=fig2) 

    ax2[1,1].grid(lw=2)
    
    # Force the colorbar to be the same size as the axes
    divider00 = make_axes_locatable(ax2[0,0])
    cax00 = divider00.append_axes("right", size="5%")

    divider01 = make_axes_locatable(ax2[0,1])
    cax01 = divider01.append_axes("right", size="5%")

    divider10 = make_axes_locatable(ax2[1,0])
    cax10 = divider10.append_axes("right", size="5%")  
    
    fig2.colorbar(im00,cax=cax00); fig2.colorbar(im01,cax=cax01); fig2.colorbar(im10,cax=cax10)
    ax2[0,0].set_title('Original (noisy) image'); ax2[0,1].set_title('Reconstructed image - Frac. of energy = '+str(np.round(recovered_energy_fraction,4)))
    ax2[1,0].set_title('Residual image - Frac. of energy = '+str(np.round(residual_energy_fraction,4))); 
    fig2.suptitle('Sparse decomposition from an semi-intelligent Dictionary')

    if (noise_scale == 0):
        ax2[1,1].set_title('Rel. magnitude of coefficients - '+str(n_nonzero_coefs))
    else:
        ax2[1,1].set_title('Rel. diff in magnitude ' \
                + r'$\displaystyle \frac{N.C - O.C}{\left\lVert O.C \right\rVert_2}$'\
                + ' - ' + str(n_nonzero_coefs))
    
    fig2.tight_layout()
    plt.savefig(Path, dpi=200)
    plt.clf()

def sparse_solver(D, signal, N1,N2, Num_of_shapelets = None):

    """ Find appropriate weights for the basis coefficients 
    obtained by the inner product routine in __shapelet_decomposition__
    using the Orthogonal Matching Pursuit algorithm

    @param D basis coefficient matrix; columns contain basis vectors
    @param signal original image to be decomposed into shapelet basis
    @param N1,N2 number of n and m quantum numbers respectively
    
    """
    n_nonzero_coefs = Num_of_shapelets
    if Num_of_shapelets == None:
        n_nonzero_coefs = N1*N2/4
    omp = OMP(n_nonzero_coefs)
    omp.fit(D,signal)
    sparse_coefs = omp.coef_
    sparse_idx = sparse_coefs.nonzero()
    sparse_reconst = np.dot(D,sparse_coefs)
    sparse_residual = signal - sparse_reconst

    residual_energy_fraction = np.sum(sparse_residual**2)/np.sum(signal**2)
    recovered_energy_fraction = np.sum(sparse_reconst**2)/np.sum(signal**2)

    return sparse_coefs, sparse_reconst, sparse_residual, \
            residual_energy_fraction, recovered_energy_fraction, n_nonzero_coefs

def solver_SVD(D, signal):

    """ Find appropriate coefficients for basis vectors contained in D, reconstruct image,
    calculate residual and residual and energy fraction using the Singular Value Decomposition
    
    @param D basis coefficient matrix
    @param signal original image

    """

    rows_SVD, columns_SVD = np.shape(D)
    U, s, VT = linalg.svd(D, full_matrices = True, compute_uv = True)    
  
    # In the docs it is said that the matrix returns V_transpose and not V 
    V = VT.transpose() 
    
    # Make 1 / sigma_i array, where sigma_i are the singular values obtained from SVD
    dual_s = 1./s

    # Initialize diagonal matrices for singular values
    S = np.zeros(D.shape)
    S_dual = np.zeros(D.shape)
    
    # Put singular values on the diagonal
    for i in xrange(len(s)):
        S[i,i] = s[i]
        S_dual[i,i] = dual_s[i]
    
    coeffs_SVD = np.dot(V, np.dot(S_dual.transpose(), np.dot(U.transpose(),signal)))
    
    n_nonzero_coefs_SVD = np.count_nonzero(coeffs_SVD)
    reconstruction_SVD = np.dot(D,coeffs_SVD)
    residual_SVD = signal - reconstruction_SVD
    residual_energy_fraction_SVD = np.sum(residual_SVD**2)/np.sum(signal**2)
    recovered_energy_fraction_SVD = np.sum(reconstruction_SVD**2)/np.sum(signal**2)
    
    return coeffs_SVD, reconstruction_SVD, residual_SVD, \
            residual_energy_fraction_SVD, recovered_energy_fraction_SVD, n_nonzero_coefs_SVD

def solver_lstsq(D, signal):

    """Find appropriate coefficients for the basis matrix D, reconstruct the image, calculate
    residual and energy and residual fraction using the Orthogonal Matching Pursuit algorithm
    
    @param D basis coefficient matrix
    @param signal original image
    """
    
    coeffs_lstsq = linalg.lstsq(D, signal)[0] # For soe reason this is not he right shape? 
    n_nonzero_coefs_lstsq = np.count_nonzero(coeffs_lstsq)
    if(DEBUG):
        print(np.shape(coeffs_lstsq), np.shape(D))
        print(coeffs_lstsq)
    reconstruction_lstsq = np.dot(D,coeffs_lstsq)
    residual_lstsq = signal - reconstruction_lstsq
    residual_energy_fraction_lstsq = np.sum(residual_lstsq**2)/np.sum(signal**2)
    recovered_energy_fraction_lstsq = np.sum(reconstruction_lstsq**2)/np.sum(signal**2)
    
    return coeffs_lstsq, reconstruction_lstsq, residual_lstsq, \
            residual_energy_fraction_lstsq, recovered_energy_fraction_lstsq, n_nonzero_coefs_lstsq

def solver_lasso_reg(D, signal, alpha = None):
    
    """Find appropriate basis coefficients for the vectors inside basis matrix D
    reconstruct the image, calculate residual and energy and residual fraction with 
    the Lasso regularization technique minimizing the L_1 norm
    
    @param D basis coefficient matrix
    @param signal original image

    """
    if (alpha == None):
        alpha = 0.001
    
    lasso_fit = linear_model.Lasso(alpha = alpha, max_iter=10000, fit_intercept = False).fit(D, signal)
    coeffs_lasso = lasso_fit.coef_
    reconstruction_lasso = np.dot(D, coeffs_lasso)
    residual_lasso = signal - reconstruction_lasso
    residual_energy_fraction_lasso = np.sum(residual_lasso**2)/np.sum(signal**2)
    recovered_energy_fraction_lasso = np.sum(reconstruction_lasso**2)/np.sum(signal**2)
    n_nonzero_coefs_lasso = np.count_nonzero(coeffs_lasso)

    return coeffs_lasso, reconstruction_lasso, residual_lasso, \
            residual_energy_fraction_lasso, recovered_energy_fraction_lasso, n_nonzero_coefs_lasso

def shapelet_decomposition(N1=20,N2=20, basis = 'XY', solver = 'sparse', image = None, \
        coeff_0 = None, noise_scale = None, alpha_ = None, Num_of_shapelets = None, n_max = None):

    """ Do the shapelet decomposition
    
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
    """

    # Obtaining galaxy images
    if (image == None) or (noise_scale == 0):
        cube = pyfits.getdata('../../data/cube_real.fits')
        cube_noiseless = pyfits.getdata('../../data/cube_real_noiseless.fits')
        background = 1.e6*0.16**2
        img = galsim.Image(78,78) # cube has 100, 78x78 images
        size_X = 78; size_Y = 78
        pick_an_img = [91]
    else:
        # Added this part for stability test with noise
        # Background is zero because I already substracted background in the first decomp
        size_X = np.shape(image)[0]; size_Y = np.shape(image)[1]
        noise_img = noise_scale * np.random.rand(size_X, size_Y)
        image = image + noise_img
        cube = [image]
        background = 0
        pick_an_img = [0] 
    
    if (basis == 'XY') or (basis == 'Elliptical'):
        D = np.zeros((size_X*size_Y,4*N1*N2)) # alloc for Dictionary
        base_coefs = np.zeros((N1,N2))
    elif (basis == 'Polar'):
        D = np.zeros((size_X*size_Y,4*N1*N2))#, dtype=complex)
        base_coefs = np.zeros((N1,N2))#, dtype=complex)

    X = np.linspace(0,size_X-1,size_X)  
    Y = np.linspace(0,size_Y-1,size_Y)
    
    for img_idx in pick_an_img:

        #Take the image, reduce for background, find moments set n_max

        cube[img_idx] -= background
        img = galsim.Image(cube[img_idx],xmin=0,ymin=0)
        shape = img.FindAdaptiveMom()
        x0,y0 = shape.moments_centroid.x, shape.moments_centroid.y ## possible swap b/w x,y
        sigma = shape.moments_sigma
        if n_max == None:
            n_max = 20

        #Make a meshgrid for the polar shapelets

        Xv, Yv = np.meshgrid((X-x0),(Y-y0))
        R = np.sqrt(Xv**2 + Yv**2)
        
        Phi = np.zeros_like(R)
        for i in xrange(np.shape(Xv)[0]):
            for j in xrange(np.shape(Xv)[1]):
                Phi[i,j] = math.atan2(Yv[i,j], Xv[i,j])

        signal = cube[img_idx].flatten() 

        shapelet_reconst = np.zeros_like(signal)

        #Decompose into Polar or XY or Elliptical w/ inner product 
        if (basis == 'Polar'):

            #Set the counter for columns of basis matrix D

            k_p = 0
            polar_basis = 'refregier' 

            for n in xrange(N1):
                for m in xrange(-n,n+1,2):
                    
                    # n_max - defined as:
                    # theta_max (galaxy size) / theta_min (smallest variation size) - 1 
                    
                    if (n <= n_max): 
                        if (polar_basis == 'refregier'):
                            arr_res = \
                                    p_shapelet.polar_shapelets_refregier(n,m,sigma)(R,Phi).flatten() 
                            arr = arr_res
                            arr_im = arr_res.imag
                        elif (polar_basis == 'berry'):
                            arr_res = p_shapelet.polar_shapelets_berry(n,m,sigma)(R,Phi).flatten()
                            arr = arr_res 
                            arr_im = arr_res.imag
                        # Make the basis matrix D
                        D[:,k_p] = arr_res 
                        k_p += 1
                        
                        # Calculate the norms of basis vectors and coefficients
                        arr_norm2_res = np.dot(arr_res,arr_res)
                        arr_norm2 = np.dot(arr, arr)
                        arr_norm_im2 = np.dot(arr_im, arr_im)
                        coef_r = np.dot(arr,signal)
                        coef_im = np.dot(arr_im, signal)
                        coef_res= np.dot(arr_res, signal)

                        # Add coefficients to basis_coefs array for later use
                        # Make the shapelet reconstruction
                        if (coef_im==0) or (coef_r==0): 
                            base_coefs[n,m]=0 
                        else: 
                            base_coefs[n,m] = coef_res/np.sqrt(arr_norm2_res)
                            shapelet_reconst = shapelet_reconst \
                                    + coef_res*arr_res/arr_norm2_res
                    else: break
        elif(basis == 'XY'):

             for k in xrange(N1*N2):
                m,n = k/N1, k%N1 
                if (m+n <= n_max): 
                    
                    arr = shapelet2d(m,n,x0=x0,y0=y0,sx=sigma,sy=sigma)(X,Y).flatten() 
                    D[:,k] = arr
                    arr_norm2 = np.dot(arr, arr)
                    coef = np.dot(arr,signal)
                    if(coef==0): 
                        base_coefs[n,m] = 0
                    else: 
                        base_coefs[n,m] = coef/np.sqrt(arr_norm2)                        
                        shapelet_reconst = shapelet_reconst + (coef*arr)/arr_norm2 
                else: break
        elif(basis == 'Elliptical'):
            
            pass

        residual= signal - shapelet_reconst
        residual_energy_fraction = np.sum(residual**2)/np.sum(signal**2)
        recovered_energy_fraction = np.sum(shapelet_reconst**2)/np.sum(signal**2)

        print "Comparing moments_amp to base_coefs[0,0]", np.abs(base_coefs[0,0]), shape.moments_amp
        print "Base coefficients sum over signal", \
                (np.sum(base_coefs**2))/(np.sum(signal**2)), \
                (np.sum(residual**2)/np.sum(signal**2)) 

        # Make the strings for nice representation in the output
        noise_scale_str = str("%.3e" % (noise_scale))
        alpha_str = str("%.3e" % (alpha_))

        mkdir_p('Plots/Decomp/')
        plot_decomposition(cube, img_idx, size_X, size_Y, \
                base_coefs, N1, N2, shapelet_reconst, signal, \
                residual, residual_energy_fraction ,recovered_energy_fraction, \
                'Plots/Decomp/'+ solver+ '_' + basis +'_'+noise_scale_str +'_'\
                +str(n_max) + '_.png')

        # Sparse solver
        if (solver == 'sparse'):
            
            if Num_of_shapelets == None:
                Num_of_shapelets = 10

            sparse_coefs, sparse_reconst, sparse_residual, \
                residual_energy_fraction, recovered_energy_fraction, \
                n_nonzero_coefs = sparse_solver(D, signal, N1,N2,Num_of_shapelets)

            if (noise_scale == 0):
                sparse_coefs_plot = sparse_coefs
            else:
                sparse_coefs_plot = (sparse_coefs - coeff_0)/linalg.norm(coeff_0)

            # Make a dir for storage of decompositions 
            mkdir_p('Plots/Sparse/'+str(Num_of_shapelets) + '/')

            plot_solution(N1,N2,cube,img_idx,size_X,size_Y, \
                    sparse_reconst, sparse_residual, sparse_coefs,\
                    recovered_energy_fraction, residual_energy_fraction, n_nonzero_coefs, \
                    noise_scale, \
                    'Plots/Sparse/'+ str(Num_of_shapelets) + '/Sparse_solution_'\
                    +noise_scale_str+'_'+str(N1)+'_'+str(N2)+'_'+str(n_max)+'_'+basis+'_'\
                    +str(Num_of_shapelets)+'_.png')

            return sparse_reconst.reshape(size_X,size_Y), sparse_coefs   

        # SVD solver // following berry approach
        if (solver == 'SVD'):
            
            coeffs_SVD, reconstruction_SVD, residual_SVD, \
            residual_energy_fraction_SVD, recovered_energy_fraction_SVD, \
            n_nonzero_coefs_SVD = solver_SVD(D, signal)
            
            if (noise_scale == 0):
                SVD_coefs_plot = coefs_SVD
            else:
                SVD_coefs_plot = (coefs_SVD - coeff_0)/linalg.norm(coeff_0)

            # Make a dir for storage of decomp.
            mkdir_p('Plots/SVD/')

            plot_solution(N1,N2,cube, img_idx,size_X,size_Y, \
                    reconstruction_SVD, residual_SVD, SVD_coefs_plot, \
                    recovered_energy_fraction_SVD, residual_energy_fraction_SVD, \
                    n_nonzero_coefs_SVD, noise_scale, \
                    'Plots/SVD/SVD_solution_' \
                     +noise_scale_str+'_'+str(N1)+'_'+str(N2)+'_'+str(n_max)+'_'+basis+'_.png')
            
            return reconstruction_SVD.reshape(size_X,size_Y), coeffs_SVD
            

        #Ordinary least squares solver
        if (solver == 'lstsq'):  

            coeffs_lstsq, reconstruction_lstsq, residual_lstsq, \
            residual_energy_fraction_lstsq, recovered_energy_fraction_lstsq, \
            n_nonzero_coefs_lstsq = solver_lstsq(D, signal)
            
            if (noise_scale == 0):
                lstsq_coefs_plot = coeffs_lstsq 
            else:
                lstsq_coefs_plot = (coeffs_lstsq - coeff_0)/linalg.norm(coeff_0)

            # Make a dir for storage of decomp.
            mkdir_p('Plots/Lstsq/')

            plot_solution(N1,N2,cube, img_idx,size_X,size_Y,  \
                    reconstruction_lstsq, residual_lstsq, \
                    lstsq_coefs_plot, \
                    recovered_energy_fraction_lstsq, residual_energy_fraction_lstsq, \
                    n_nonzero_coefs_lstsq, noise_scale, \
                    'Plots/Lstsq/lstsq_solution_'\
                    +noise_scale_str+'_'+str(N1)+'_'+str(N2)+'_'+str(n_max)+'_'+basis+'_.png')
            
            
            return reconstruction_lstsq.reshape(size_X,size_Y), coeffs_lstsq
        
        if (solver == 'lasso'): #This is with the Lasso regularization
            
            if alpha_ == None:
                alpha_ = 0.0001
                
            coeffs_lasso, reconstruction_lasso, residual_lasso, \
                residual_energy_fraction_lasso, recovered_energy_fraction_lasso, \
                n_nonzero_coefs_lasso = solver_lasso_reg(D, signal, alpha_)
        
            if (noise_scale == 0):
                lasso_coefs_plot = coeffs_lasso
            else:
                lasso_coefs_plot = (coeffs_lasso - coeff_0)/linalg.norm(coeff_0)
                
            # Make a dir for storage of decomp.
            mkdir_p('Plots/Lasso/'+alpha_str+'/')
            
            plot_solution(N1,N2,cube, img_idx,size_X,size_Y,  \
                        reconstruction_lasso, residual_lasso, \
                        lasso_coefs_plot, \
                        recovered_energy_fraction_lasso, residual_energy_fraction_lasso, \
                        n_nonzero_coefs_lasso, noise_scale, \
                        'Plots/Lasso/' + alpha_str + '/lasso_solution_'\
                        +noise_scale_str+ '_' + str(N1)+'_'+str(N2)+'_'+str(n_max)+'_'+basis+'_'\
                        +alpha_str+'_.png')

            return reconstruction_lasso.reshape(size_X, size_Y), coeffs_lasso

def plot_stability(coeff_stability, coeff_0, N1, N2, noise_scale, \
        fname_imag = 'Plots/Lasso/Stability/',\
        variance_file = 'Plots/Lasso/Stability/',\
        mid_word = None):
    
    # Find the mean coefficients
    coeff_stability = (coeff_stability/len(noise_scale) - coeff_0)\
            /linalg.norm(coeff_0)

    image_stability = coeff_stability.reshape((2*N1,2*N2))

    fig, ax = plt.subplots()
    
    coefs = coeff_stability.reshape(2*N1,2*N2)
    coeff_plot2d(coefs,N1*2,N2*2,ax=ax,fig=fig) 
    
    # Add LaTeX parsing
    #plt.rc('text', usetex=True)
    #plt.rc('font', **{'family' : "sans-serif"})
    #params = {'text.latex.preamble' : [r'\usepackage{siunitx}', \
    #            r'\usepackage[utf8]{inputenc}', r'\usepackage{amsmath}']}
    #plt.rcParams.update(params)
    
    ax.grid(lw=2)
    ax.set_title('Stability of coefs '\
            + r'$\displaystyle \frac{<N.C> - O.C}{\left\lVert O.C \right\rVert_2}$' \
            + '\n' \
            + 'N.C is averaged over the number of noise realizations')

    mkdir_p(fname_imag)
    
    fig.tight_layout()
    plt.savefig(fname_imag + 'stability_'+mid_word+'_.png', dpi = 200)
    plt.clf()

    # Calculate the variance of decompositions
    variance = np.var(\
            (coeff_stability/len(noise_scale) - coeff_0) \
            /linalg.norm(coeff_0))
    
    mkdir_p(variance_file + mid_word + '/')
    
    variance_file = open(variance_file + mid_word+'/' + 'variance.txt', 'w')
    variance_file.write("%.5f\n" % (variance))

def test_stability(solver, basis, noise_scale, alpha_ = None, Num_of_shapelets_array = None):
    
    N1 = 20; N2=20; n_max = 20; Num_of_shapelets = None; alpha = None
    
    image = None; image_curr = None; coeffs_0 = None; coeffs_curr = None; coeff_stability = None

    # Now select alphas if the method is lasso
    if solver == 'lasso':
        for l in range(len(alpha_)):
            
            alpha = alpha_[l]
            # Iterate through different noise scales
            # image - image to be decomposed
            # image_curr - temporary storage for decomposed image
            
            image, coeff_0 =\
                        shapelet_decomposition(N1,N2,basis,solver,\
                        None, None, 0,\
                        alpha,\
                        n_max = n_max)

            coeff_stability = np.zeros_like(coeff_0)
            
            for noise in noise_scale:
                image_curr, coeffs_curr =\
                        shapelet_decomposition(N1,N2,basis,solver,\
                        image, coeff_0, noise,\
                        alpha,\
                        n_max = n_max)
                coeff_stability += coeffs_curr

            plot_stability(coeff_stability, coeff_0, N1, N2, noise_scale, \
                    fname_imag = 'Plots/Lasso/Stability/',\
                    variance_file = 'Plots/Lasso/Stability/',\
                    mid_word = str("%.3e" % (alpha))+'_'+str(basis))
             
    elif(solver == 'sparse'):
        
        # Select number of shapelets that would be selected by OMP
        for c in range(len(Num_of_shapelets_array)):
            Num_of_shapelets = Num_of_shapelets_array[c]
            
            # Initialize the image for decomposition
            image, coeff_0=\
                shapelet_decomposition(N1,N2,basis,solver,\
                None, None, 0,\
                alpha,\
                Num_of_shapelets, n_max)
            
            coeff_stability = np.zeros_like(coeff_0)
            # Iterate through different noise scales
            # image - image to be decomposed
            # image_curr - temporary storage for decomposed image
            for noise in noise_scale:
                image_curr, coeffs_curr =\
                        shapelet_decomposition(N1,N2,basis,solver,\
                        image, coeff_0, noise,\
                        Num_of_shapelets = Num_of_shapelets, n_max = n_max)
                coeff_stability += coeffs_curr
            
            plot_stability(coeff_stability, coeff_0, N1, N2, noise_scale, \
                    fname_imag = 'Plots/Sparse/Stability/',\
                    variance_file = 'Plots/Sparse/Stability/',\
                    mid_word = str(Num_of_shapelets) + '_' + str(basis))         
            
    elif(solver == 'lstsq'):
        image, coeff_0 =\
                shapelet_decomposition(N1,N2,basis,solver,\
                None, None, 0,\
                alpha,\
                Num_of_shapelets, n_max)

        coeff_stability = np.zeros_like(coeff_0)
        # Iterate through different noise scales
        # image - image to be decomposed
        # image_curr - temporary storage for decomposed image
        for noise in noise_scale:
            image_curr, coeffs_curr =\
                    shapelet_decomposition(N1,N2,basis,solver,\
                    image, coeff_0, noise,\
                    n_max = n_max)
            coeff_stability += coeffs_curr
        
        plot_stability(coeff_stability, coeff_0, N1, N2, noise_scale, \
                fname_imag = 'Plots/Lstsq/Stability/',\
                variance_file = 'Plots/Lstsq/Stability/',\
                mid_word = str(n_max) + '_' + str(basis))

if __name__=='__main__':
    
    Num_of_shapelets_array = [5,10,20]
    methods = ['lasso', 'sparse', 'lstsq']
    noise_scale = np.logspace(1, 2.5, 5)
    alpha_ = np.logspace(-5,-1,6)
    basis_array = ['Polar', 'XY']
    
    # Select a method for fitting the coefficients
    for basis in basis_array:
        
        for solver in ['lasso']:#range(len(methods)):

            test_stability(solver, basis, noise_scale,\
                    alpha_ = alpha_, Num_of_shapelets_array = Num_of_shapelets_array)
           
    #show_some_shapelets()
    #p_shapelet.plot_shapelets(6,2,1)
    #check_orthonormality()
