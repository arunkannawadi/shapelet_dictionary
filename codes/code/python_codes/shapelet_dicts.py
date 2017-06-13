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

#import pdb; pdb.set_trace()

#------------------
# berry == Berry et al. MNRAS 2004
#------------------

from sklearn.linear_model import OrthogonalMatchingPursuit as OMP
import polar_shapelet_decomp as p_shapelet

DEBUG =1

##Define orthonormal basis - shapelets
def shapelet1d(n,x0=0,s=1):
    def sqfac(k):
        fac = 1.
        for i in xrange(k):
            fac *= np.sqrt(i+1)
        return fac

    u = lambda x: (x-x0)/s
    fn = lambda x: (1./(2*np.pi)**0.25)*(1./sqfac(n))*hermitenorm(n)(u(x))*np.exp(-0.25*u(x)**2) 
    return fn

def shapelet2d(m,n,x0=0,y0=0,sx=1,sy=1):
    u = lambda x: (x-x0)/sx
    v = lambda y: (y-y0)/sy
    fn = lambda x,y: np.outer(shapelet1d(m)(u(x)),shapelet1d(n)(v(y)))
    return fn

def elliptical_shapelet(m,n,x0=0,y0=0,sx=1,sy=1,theta=0):
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
    if ax is None:
      fig, ax = plt.subplots()

    ### INPUT VALIDATION NEEDS TO BE FILLED IN

    coeffs_reshaped = np.abs(coeffs.reshape(N1,N2)) ## does not 
    coeffs_reshaped /= coeffs_reshaped.max()
    im = ax.imshow(coeffs_reshaped,cmap=cm.bwr,interpolation='none')
    fig.colorbar(im,ax=ax,orientation=orientation)
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

def plot_decomposition(cube_real, img_idx, base_coefs,N1,N2,shapelet_reconst, signal, residual,\
        residual_energy_fraction ,recovered_energy_fraction, basis):

    fig, ax = plt.subplots(2,2)
    coeff_plot2d(base_coefs,N1,N2,ax=ax[1,1],fig=fig)
    vmin, vmax = min(shapelet_reconst.min(),signal.min()), max(shapelet_reconst.max(),signal.max())

    im00 = ax[0,0].imshow(cube_real[img_idx],vmin=vmin,vmax=vmax)
    im01 = ax[0,1].imshow(shapelet_reconst.reshape(78,78),vmin=vmin,vmax=vmax)
    im10 = ax[1,0].imshow(residual.reshape(78,78))
    fig.colorbar(im00,ax=ax[0,0]); fig.colorbar(im01,ax=ax[0,1]); fig.colorbar(im10,ax=ax[1,0])
    ax[0,0].set_title('Original (noisy) image')
    ax[0,1].set_title('Reconstructed image - Frac. of energy = '\
            +str(np.round(recovered_energy_fraction,4)))
    ax[1,0].set_title('Residual image - Frac. of energy = '\
            +str(np.round(residual_energy_fraction,4)))
    ax[1,1].set_title('Rel. magnitude of coefficients')
    fig.suptitle('Shapelet Basis decomposition')
    
    def on_draw(event):
       bboxes = []
       for label in labels:
           bbox = label.get_window_extent()
           # the figure transform goes from relative coords->pixels and we
           # want the inverse of that
           bboxi = bbox.inverse_transformed(fig.transFigure)
           bboxes.append(bboxi)

       # this is the bbox that bounds all the bboxes, again in relative
       # figure coords
       bbox = mtransforms.Bbox.union(bboxes)
       if fig.subplotpars.left < bbox.width:
           # we need to move it over
           fig.subplots_adjust(left=1.1*bbox.width) # pad a little
           fig.canvas.draw()
       return False

    fig.canvas.mpl_connect('draw_event', on_draw)
    
    plt.show()

    if(basis == 'Polar'):
        plt.savefig('Decomp_cartesian.png')
    elif(basis == 'XY'):
        plt.savefig('Decomp_Polar.png')
    elif(basis == 'Elliptical'):
        plt.savefig('Decomp_Elliptical.png')

    return fig

def plot_solution(N1,N2,cube_real, img_idx, reconst, residual, coefs,\
                recovered_energy_fraction, residual_energy_fraction, n_nonzero_coefs, fig, Path):
    
    fig2, ax2 = plt.subplots(2,2)
    im00 = ax2[0,0].imshow(cube_real[img_idx])
    im01 = ax2[0,1].imshow(reconst.reshape(78,78))
    im10 = ax2[1,0].imshow(residual.reshape(78,78))
    print coefs.shape
    coefs = coefs.reshape(2*N1,2*N2)
    coeff_plot2d(coefs,N1*2,N2*2,ax=ax2[1,1],fig=fig) 

    ax2[1,1].grid(lw=2)
    fig2.colorbar(im00,ax=ax2[0,0]); fig2.colorbar(im01,ax=ax2[0,1]); fig2.colorbar(im10,ax=ax2[1,0])
    ax2[0,0].set_title('Original (noisy) image'); ax2[0,1].set_title('Reconstructed image - Frac. of energy = '+str(np.round(recovered_energy_fraction,4)))
    ax2[1,0].set_title('Residual image - Frac. of energy = '+str(np.round(residual_energy_fraction,4))); ax2[1,1].set_title('Rel. magnitude of coefficients - '+str(n_nonzero_coefs))
    fig2.suptitle('Sparse decomposition from an semi-intelligent Dictionary :) ')

    plt.savefig(Path)


def sparse_solver(D, signal, N1, N2):
   
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
    
    rows_SVD, columns_SVD = np.shape(D)
    U, s, VT = linalg.svd(D, full_matrices = True, compute_uv = True)
    
    if (DEBUG):
        print(s); print(len(s))

    V = VT.transpose() # In the docs it is said that the matrix returns V_transpose and not V 
   
    dual_s = 1./s
    S = np.eye(rows_SVD, columns_SVD)*s; S_dual = np.eye(rows_SVD, columns_SVD)*dual_s

    if(DEBUG):
        print(S); print("S_dual:"); print(S_dual)
        print(np.shape(S), np.shape(S_dual))
        print("U: ", np.shape(U),"V: ", np.shape(V),"S_dual: ", np.shape(S_dual), "signal: ", np.shape(signal))

    coeffs_SVD = np.dot(V, np.dot(S_dual.transpose(), np.dot(U.transpose(),signal)))
    
    if(DEBUG):
        print('shape_coeffs ', np.shape(coeffs_SVD))
        print('shape_D', np.shape(D))
    n_nonzero_coefs_SVD = np.count_nonzero(coeffs_SVD)
    reconstruction_SVD = np.dot(D,coeffs_SVD)
    residual_SVD = signal - reconstruction_SVD
    residual_energy_fraction_SVD = np.sum(residual_SVD**2)/np.sum(signal**2)
    recovered_energy_fraction_SVD = np.sum(reconstruction_SVD**2)/np.sum(signal**2)
    
    return coeffs_SVD, reconstruction_SVD, residual_SVD, \
            residual_energy_fraction_SVD, recovered_energy_fraction_SVD, n_nonzero_coefs_SVD

def solver_lstsq(D, signal):
    
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

def solver_lasso_reg(D, signal):
    
    lasso_fit = linear_model.Lasso(alpha = 0.001, max_iter=10000, fit_intercept = False).fit(D, signal)
    coeffs_lasso = lasso_fit.coef_
    reconstruction_lasso = np.dot(D, coeffs_lasso)
    residual_lasso = signal - reconstruction_lasso
    residual_energy_fraction_lasso = np.sum(residual_lasso**2)/np.sum(signal**2)
    recovered_energy_fraction_lasso = np.sum(reconstruction_lasso**2)/np.sum(signal**2)
    n_nonzero_coefs_lasso = np.count_nonzero(coeffs_lasso)

    return coeffs_lasso, reconstruction_lasso, residual_lasso, \
            residual_energy_fraction_lasso, recovered_energy_fraction_lasso, n_nonzero_coefs_lasso


def shapelet_decomposition(N1=20,N2=20, basis = 'XY', solver = 'sparse', noise = False):
    # Obtaining galaxy images
    cube_real = pyfits.getdata('../../data/cube_real.fits')
    cubr_real_noiseless = pyfits.getdata('../../data/cube_real_noiseless.fits')
    background = 1.e6*0.16**2
    img = galsim.Image(78,78) # cube_real has 100, 78x78 images
    D = np.zeros((78*78,4*N1*N2)) # alloc for Dictionary
    base_coefs = np.zeros((N1,N2))
    X = np.linspace(0,77,78)  
    Y = np.linspace(0,77,78)
    
    for img_idx in [91]:
        cube_real[img_idx] -= background
        img = galsim.Image(cube_real[img_idx],xmin=0,ymin=0)
        shape = img.FindAdaptiveMom()
        x0,y0 = shape.moments_centroid.x, shape.moments_centroid.y ## possible swap b/w x,y
        sigma = shape.moments_sigma
        
        Xv, Yv = np.meshgrid((X-x0),(Y-y0))
        R = np.sqrt(Xv**2 + Yv**2)
        
        Phi = np.zeros_like(R)
        for i in xrange(np.shape(Xv)[0]):
            for j in xrange(np.shape(Xv)[1]):
                Phi[i,j] = math.atan2(Yv[i,j], Xv[i,j])

        signal = cube_real[img_idx].flatten()

        if(1):
            import random
            random.seed()
            print(signal, np.shape(signal))
            signal = signal + np.random.rand(np.shape(signal)[0])
            print(signal, np.shape(signal))

        shapelet_reconst = np.zeros_like(signal)

        #Decompose into Polar or XY 
        if (basis == 'Polar'):
            k_p = 0
            for n in xrange(N1):
                for m in xrange(-n,n+1,2):
                    if (n <= (78/sigma - 1)): # n_max ~ theta_max (image size) / theta_min (pixel or kernel smoothing size) -1 
                        arr = p_shapelet.polar_shapelets_real(n,m,sigma)(R, Phi).flatten() 
                        arr_im = p_shapelet.polar_shapelets_imag(n,m,sigma)(R, Phi).flatten()
                        
                        #I'M ONLY LETTING REAL COMPONENTS GO TO THE D MATRIX
                        D[:,k_p] = arr; #D[:,k+N1*N2]=arr2; D[:,k+2*N1*N2]=arr3; D[:,k+3*N1*N2]=arr4
                        k_p += 1
                        arr_norm2 = np.dot(arr, arr)
                        arr_norm_im2 = np.dot(arr_im, arr_im)
                        coef_r = np.dot(arr,signal)
                        coef_im = np.dot(arr_im, signal)
                        if (coef_im==0) or (coef_r==0): 
                            base_coefs[n,m] = 0
                        else: 
                            base_coefs[n,m] = coef_r/np.sqrt(arr_norm2)                            
                            shapelet_reconst = shapelet_reconst + coef_r*arr/arr_norm2 + coef_im*arr_im/arr_norm_im2
                    else: break
        elif(basis == 'XY'):
             for k in xrange(N1*N2):
                m,n = k/N1, k%N1 
                if (m+n <= (78/sigma - 1)): 
                    arr = shapelet2d(m,n,x0=x0,y0=y0,sx=sigma,sy=sigma)(X,Y).flatten()
                    #arr2 = shapelet2d(m,n,x0=x0,y0=y0,sx=0.5*sigma,sy=0.5*sigma)(X,Y).flatten()
                    #arr3 = shapelet2d(m,n,x0=x0,y0=y0,sx=1.5*sigma,sy=2.*sigma)(X,Y).flatten()
                    #arr4 = shapelet2d(m,n,x0=x0,y0=y0,sx=2.0*sigma,sy=2.0*sigma)(X,Y).flatten()

                    D[:,k] = arr; #D[:,k+N1*N2]=arr2; D[:,k+2*N1*N2]=arr3; D[:,k+3*N1*N2]=arr4
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

        print "Comparing moments_amp to base_coefs[0,0]", base_coefs[0,0], shape.moments_amp
        print "Base coefficients sum over signal", np.sum(base_coefs**2)/(np.sum(signal**2)), np.sum(residual**2)/np.sum(signal**2)

        fig = plot_decomposition(cube_real, img_idx, base_coefs,N1,N2,shapelet_reconst, signal, residual,\
                residual_energy_fraction ,recovered_energy_fraction, basis)         


        # Sparse solver
        if (solver == 'sparse'):
            sparse_coefs, sparse_reconst, sparse_residual, \
                residual_energy_fraction_sparse, recovered_energy_fraction_sparse, \
                n_nonzero_coefs = sparse_solver(D, signal, N1, N2)
            plot_solution(N1,N2,cube_real, img_idx, sparse_reconst, sparse_residual, sparse_coefs,\
                    recovered_energy_fraction_sparse, residual_energy_fraction_sparse, \
                    n_nonzero_coefs, \
                    fig, 'Sparse_solution.png')

        # SVD solver // following berry approach
        if (solver == 'SVD'):

            coeffs_SVD, reconstruction_SVD, residual_SVD,\
            residual_energy_fraction_SVD, recovered_energy_fraction_SVD, \
            n_nonzero_coefs_SVD =solver_SVD(D, signal)
            
            print(np.shape(residual_SVD), np.shape(reconstruction_SVD))
            plot_solution(N1,N2,cube_real, img_idx, reconstruction_SVD, residual_SVD, coeffs_SVD, \
                    recovered_energy_fraction_SVD, residual_energy_fraction_SVD, \
                    n_nonzero_coefs_SVD,\
                    fig, 'SVD_solution.png')

        #Ordinary least squares solver
        if (solver == 'P_2'):
            
            coeffs_lstsq, reconstruction_lstsq, residual_lstsq, \
            residual_energy_fraction_lstsq, recovered_energy_fraction_lstsq, \
            n_nonzero_coefs_lstsq = solver_lstsq(D, signal)
            
            plot_solution(N1,N2,cube_real, img_idx, reconstruction_lstsq, residual_lstsq, \
                    coeffs_lstsq, recovered_energy_fraction, residual_energy_fraction, \
                    n_nonzero_coefs_lstsq, fig, 'lstsq_solution.png')


        if (solver == 'P_1'): #This is with the Lasso regularization
            coeffs_lasso, reconstruction_lasso, residual_lasso, \
            residual_energy_fraction_lasso, recovered_energy_fraction_lasso, \
            n_nonzero_coefs_lasso = solver_lasso_reg(D, signal)
            
            plot_solution(N1,N2,cube_real, img_idx, reconstruction_lasso, residual_lasso, \
                    coeffs_lasso, recovered_energy_fraction_lasso, residual_energy_fraction_lasso, \
                    n_nonzero_coefs_lasso, fig, 'lasso_solution.png')

if __name__=='__main__':
    shapelet_decomposition(int(sys.argv[1]),int(sys.argv[2]), sys.argv[3], sys.argv[4])
    #show_some_shapelets()
    #p_shapelet.plot_shapelets(6,2,1)
    #check_orthonormality()



