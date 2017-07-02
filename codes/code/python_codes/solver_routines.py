import numpy as np
import numpy.linalg as linalg

from sklearn.linear_model import OrthogonalMatchingPursuit as OMP
from sklearn import linear_model

from plotting_routines import plot_solution
from utils.I_O_utils import *
from utils.shapelet_utils import *

#import pdb; pdb.set_trace()

def asses_diff(new_coefs, old_coefs):
    """
    Calculate the relative difference in coefficients
    """
    diff_arr = np.zeros_like(old_coefs)
    for i in xrange(len(old_coefs)):
        if old_coefs[i] != 0:
            diff_arr[i] = np.abs(new_coefs[i]/old_coefs[i] - 1.)
        else:
            diff_arr[i] = 0
    return diff_arr

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
    omp = OMP(n_nonzero_coefs = n_nonzero_coefs)
    omp.fit(D,signal)
    sparse_coefs = omp.coef_
    sparse_idx = sparse_coefs.nonzero()
    sparse_reconst = np.dot(D,sparse_coefs)
    sparse_residual = signal - sparse_reconst

    residual_energy_fraction = np.sum(sparse_residual**2)/np.sum(signal**2)
    recovered_energy_fraction = np.sum(sparse_reconst**2)/np.sum(signal**2)

    return sparse_coefs, sparse_reconst, sparse_residual, \
            residual_energy_fraction, recovered_energy_fraction, n_nonzero_coefs

def solver_SVD(D, n_nonzero, signal):

    """ Find appropriate coefficients for basis vectors contained in D, reconstruct image,
    calculate residual and residual and energy fraction using the Singular Value Decomposition
    
    @param D basis coefficient matrix
    @param signal original image

    """

    rows_SVD, columns_SVD = np.shape(D)
    U, s, VT = linalg.svd(D, full_matrices = True)    
  
    ## Count in only n_nonzero signular values set rest to zero
    s[n_nonzero:] = 0 
    
    ## In the docs it is said that the matrix returns V_transpose and not V 
    V = VT.transpose() 
    
    ## Initialize diagonal matrices for singular values
    S = np.zeros(D.shape)
    S_dual = np.zeros(D.shape)

    ## Put singular values on the diagonal
    for i in xrange(n_nonzero):
        S[i,i] = s[i]
        S_dual[i,i] = 1./s[i]
   
    coeffs_SVD = np.dot(V, np.dot(S_dual.transpose(), np.dot(U.transpose(),signal)))
    
    ## There are some residual values from V which are a lot smaller then
    ## the chosen n_nonzero coeffs, so I just set them to zero
    coeffs_SVD_r = np.zeros_like(coeffs_SVD)
    ## Get indices of n_nonzero largest by absolute value
    idx_n_nonzero = np.abs(coeffs_SVD).argsort()[-n_nonzero:][::-1]
    ## Store them in the resultant coeff array
    coeffs_SVD_r[idx_n_nonzero] = coeffs_SVD[idx_n_nonzero]

    n_nonzero_coefs_SVD = np.count_nonzero(coeffs_SVD_r)
    reconstruction_SVD = np.dot(D,coeffs_SVD_r)
    residual_SVD = signal - reconstruction_SVD
    residual_energy_fraction_SVD = np.sum(residual_SVD**2)/np.sum(signal**2)
    recovered_energy_fraction_SVD = np.sum(reconstruction_SVD**2)/np.sum(signal**2)
    
    return coeffs_SVD_r, reconstruction_SVD, residual_SVD, \
            residual_energy_fraction_SVD, recovered_energy_fraction_SVD, n_nonzero_coefs_SVD

def solver_lstsq(D, signal):

    """Find appropriate coefficients for the basis matrix D, reconstruct the image, calculate
    residual and energy and residual fraction using the Orthogonal Matching Pursuit algorithm
    
    @param D basis coefficient matrix
    @param signal original image
    """
    
    coeffs_lstsq = linalg.lstsq(D, signal)[0]  
    n_nonzero_coefs_lstsq = np.count_nonzero(coeffs_lstsq) 
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

def select_solver_do_fitting_plot(\
        f_path, \
        basis, coeff_0, noise_scale, \
        N1, N2, n_max, column_number, \
        image_initial, D, signal,solver, \
        beta_array,\
        Num_of_shapelets = 10, alpha_ = 0.0001, plot_decomp = True):
    
    ## Default for foler path
    folder_path_word = ""
    mid_path_word = ""
    end_word = ""
    mid_name_word = '_solution_'+str("%.3e" % (noise_scale))+'_'\
            +str(N1)+'_'+str(N2)\
            +'_'+str(n_max)+'_'+basis+'_'+ str(column_number) + '_'

    ## Sparse solver
    if (solver == 'sparse'):
        
        ## Include symmetric number of shapelets
        ## symmetrize the number chosen
        Num_of_shapelets = sum_max_order(basis,(get_max_order(basis,Num_of_shapelets))) 

        coeffs, reconst, residual, \
            residual_energy_fraction, recovered_energy_fraction, \
            n_nonzero_coefs = sparse_solver(D, signal, N1,N2,Num_of_shapelets)
         
        mid_path_word = str(Num_of_shapelets) + '_' + basis
        end_word = str(Num_of_shapelets)
        folder_path_word = f_path + solver + '/' + mid_path_word +'/'

    ## SVD solver // following berry approach
    elif (solver == 'svd'):
        
        coeffs, reconst, residual, \
        residual_energy_fraction, recovered_energy_fraction, \
        n_nonzero_coefs = solver_SVD(D, Num_of_shapelets, signal) 
        
        mid_path_word = str(Num_of_shapelets) + '_' + basis
        end_word = str(Num_of_shapelets)
        folder_path_word = f_path + solver + '/' + mid_path_word + '/'

    ## Ordinary least squares solver
    elif (solver == 'lstsq'):  

        coeffs, reconst, residual, \
        residual_energy_fraction, recovered_energy_fraction, \
        n_nonzero_coefs = solver_lstsq(D, signal) 

        folder_path_word = f_path + solver + '/' 
    
    elif (solver == 'lasso'): #This is with the Lasso regularization
           
        coeffs, reconst, residual, \
            residual_energy_fraction, recovered_energy_fraction, \
            n_nonzero_coefs = solver_lasso_reg(D, signal, alpha_)
    
        mid_path_word = str("%.3e" % (alpha_))
        folder_path_word = f_path + solver + '/' + mid_path_word + '/'           

        
    if (noise_scale == 0):
        coefs_plot = coeffs
    else:
        coefs_plot = asses_diff(coeffs,coeff_0)

    ## Make a dir for storage of decompositions 
    mkdir_p(folder_path_word)

    ## size_X and size_Y should be the size of the initial image
    size_X = image_initial.shape[0]
    size_Y = image_initial.shape[1]
    if plot_decomp == True:
        plot_solution(basis, N1,N2,image_initial,size_X,size_Y, \
            reconst, residual, coefs_plot,\
            recovered_energy_fraction, residual_energy_fraction, n_nonzero_coefs, \
            noise_scale, \
            f_path + solver + '/'+ mid_path_word +'/' \
            + solver + mid_name_word + end_word,\
            beta_array = beta_array)
        
    return reconst.reshape(size_X,size_Y),coeffs
