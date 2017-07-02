import galsim
import numpy as np

from shapelet_dicts import shapelet_decomposition
from plotting_routines import plot_stability

## Import the util package for the weights
from utils.get_gaussian_weight_image import get_gaussian_weight_image as gen_weight_image
from utils.I_O_utils import *
from utils.shapelet_utils import *

#import pdb; pdb.set_trace()


def do_noise_iteration(image_0,image_data,noise_img,\
        size_X,size_Y,\
        N1,N2,\
        coeff_0,label_arr,beta_arr,\
        solver,\
        noise_scale,noise_img_num,\
        Num_of_shapelets = 15, alpha = 0.0001,\
        plot_decomp = True, n_max = 21, \
        mid_word = ''):
    """
    Do the noise iterations and test the shapelet coefficients stability \
            and plot the result

    @param image_0 - Noiseless image
    @param noise_img - Matrix with gaussian noise to be added to the image_0 \
            with a factor of noise_scale
    @param size_X,size_Y - dimensions of image_0
    @param coeff_0 - Shapelet coeffs obtained for the decomposition of the image_0
    @param noise_scale - factor with which noise_img is scaled
    @param noise_img_num - Number of noise_img matrices

    """
    ## The matrix to be used for stability calculations
    ## Variance and mean difference
    coeff_stability = np.zeros((len(coeff_0),noise_img_num))
    
    ## A sample image used for the S/N calculation
    image_sample = image_0 + noise_scale*noise_img[:,0].reshape(size_X,size_Y)
    
    ## Asses the weights and calculate S/N
    weight_image,flag = gen_weight_image(image_sample)
    
    if flag == 1:
        weight_image = weight_image.flatten()
        signal_image = image_sample.flatten()

        signal_to_noise = (1./noise_scale) * np.dot(weight_image,signal_image) \
                / np.sum(weight_image)
       
        ## Make folders for storage
        f_path = 'Plots/' + str("%.3e" % (noise_scale)) + '/'
        mkdir_p(f_path)
        
        k=0
        for i in xrange(noise_img_num):
                            
            ## Add the noise_matrix to the 0 noise image
            image = image_0 + noise_scale*noise_img[:,i].reshape(size_X,size_Y)
            
            image_reconst_curr, coeffs_curr =\
                    shapelet_decomposition(image_data,\
                    f_path = f_path,\
                    N1=N1,N2=N2,basis=basis,solver=solver,\
                    image=image, coeff_0=coeff_0, noise_scale=signal_to_noise,\
                    Num_of_shapelets = Num_of_shapelets, alpha_ = alpha,\
                    plot_decomp = True,\
                    n_max = n_max,\
                    column_number = i)
            
            if coeffs_curr != None:
                coeff_stability[:,k] = coeffs_curr
                k+=1

        plot_stability(coeff_stability, coeff_0, N1, N2, noise_img_num, \
                n_max = n_max, label_arr = label_arr, \
                beta_array = beta_arr, signal_to_noise = signal_to_noise, \
                basis = basis, solver = solver, \
                path_to_save = f_path + 'Stability/',\
                mid_word = mid_word)

def test_stability(solver, basis, \
        noise_scale, noise_img = None, noise_img_num = 10,\
        size_X = 78, size_Y = 78, \
        alpha_ = None, Num_of_shapelets_array = None):

    ## Initialize values
    N1 = 20; N2=20; n_max = 21; Num_of_shapelets = None; alpha = None
    
    image = None; image_curr = None; coeffs_0 = None; coeffs_curr = None; coeff_stability = None
    
    ## Store x0,y0 (centroids) of the image and sigma
    image_data = np.zeros(3)

    ## Now select alphas if the method is lasso
    if solver == 'lasso':
        for l in xrange(len(alpha_)):
            
            alpha = alpha_[l]
            ## Iterate through different noise scales
            ## image - image to be decomposed
            ## image_curr - temporary storage for decomposed image
            
            # Make the no noise image
            f_path = 'Plots/' + str("%.3e" % (0.0)) + '/'
            mkdir_p(f_path)
            
            image_0, image_reconst, coeff_0, label_arr, beta_array  =\
                        shapelet_decomposition(image_data,\
                        f_path = f_path,\
                        N1=N1,N2=N2,basis=basis,solver=solver,\
                        image=None, coeff_0 =None, noise_scale=0,\
                        alpha_=alpha,\
                        n_max = n_max,\
                        plot_decomp = True)
            
            mid_word = str("%.3e" % (alpha))+'_'+str(basis)
            
            ## If the initial decomposition doesn't fail continue
            if (image_reconst != None):
                do_noise_iteration(image_0,image_data,noise_img,\
                        size_X,size_Y,\
                        N1,N2,\
                        coeff_0,label_arr,beta_array,\
                        solver,\
                        noise_scale,noise_img_num,\
                        Num_of_shapelets = Num_of_shapelets,alpha = alpha,\
                        plot_decomp = True, n_max = n_max,\
                        mid_word = mid_word)
                            
    elif(solver == 'sparse'):
        
        ## Select number of shapelets that would be selected by OMP
        for c in xrange(len(Num_of_shapelets_array)):
            Num_of_shapelets = Num_of_shapelets_array[c]
            
            ## Control the number of shapelets in initial decomposition for
            ## basis matrix - n_max controls the size of basis matrix

            if n_max < Num_of_shapelets:
                n_max = Num_of_shapelets

            ## Make the no noise image
            f_path = 'Plots/' + str("%.3e" % (0.0)) + '/'
            mkdir_p(f_path)

            image_0, image_reconst, coeff_0, label_arr, beta_array =\
                        shapelet_decomposition(image_data,\
                        f_path = f_path,\
                        N1=N1,N2=N2,basis=basis,solver=solver,\
                        image=None, coeff_0=None, noise_scale=0,\
                        Num_of_shapelets=Num_of_shapelets, \
                        plot_decomp = True,\
                        n_max = n_max)
            
            mid_word = str(\
                sum_max_order(basis,get_max_order(basis,Num_of_shapelets))) \
                + '_' + str(basis)
            
            ## If 0 noise decomp. fails don't do anything
            if (image_reconst != None):
                do_noise_iteration(image_0,image_data,noise_img,\
                        size_X,size_Y,\
                        N1,N2,\
                        coeff_0,label_arr,beta_array,\
                        solver,\
                        noise_scale,noise_img_num,\
                        Num_of_shapelets = Num_of_shapelets, plot_decomp = True, n_max = n_max,\
                        mid_word = mid_word)
            
    elif(solver == 'lstsq'):
        
        # Make the no noise image
        f_path = 'Plots/' + str("%.3e" % (0.0)) + '/'
        mkdir_p(f_path)

        image_0, image_reconst, coeff_0, label_arr, beta_array =\
                    shapelet_decomposition(image_data,\
                    f_path = f_path,\
                    N1=N1,N2=N2,basis=basis,solver=solver,\
                    image = None,coeff_0= None, noise_scale=0,\
                    n_max = n_max,\
                    plot_decomp = True)
        
        mid_word = str(sum_max_order(basis,get_max_order(basis,n_max)))+'_'+str(basis)
        if image_reconst!=None:
            do_noise_iteration(image_0,image_data,noise_img,\
                    size_X,size_Y,\
                    N1,N2,\
                    coeff_0,label_arr,beta_array,\
                    solver,\
                    noise_scale,noise_img_num,\
                    plot_decomp = True, n_max = n_max,\
                    mid_word = mid_word)
    elif(solver == 'svd'):
        
        for Num_of_shapelets in Num_of_shapelets_array:
            
            ## Make the no noise image
            f_path = 'Plots/' + str("%.3e" % (0.0)) + '/'
            mkdir_p(f_path)
            
            if n_max < Num_of_shapelets:
                n_max = Num_of_shapelets

            image_0, image_reconst, coeff_0, label_arr, beta_array =\
                        shapelet_decomposition(image_data,\
                        f_path = f_path,\
                        N1=N1,N2=N2,basis=basis,solver=solver,\
                        image = None,coeff_0= None, noise_scale=0,\
                        Num_of_shapelets=Num_of_shapelets,\
                        n_max = n_max,\
                        plot_decomp = True)
            
            mid_word = str(sum_max_order(basis,get_max_order(basis,n_max)))+'_'+str(basis)
            if image_reconst!=None:
                do_noise_iteration(image_0,image_data,noise_img,\
                        size_X,size_Y,\
                        N1,N2,\
                        coeff_0,label_arr,beta_array,\
                        solver,\
                        noise_scale,noise_img_num,\
                        Num_of_shapelets = Num_of_shapelets, plot_decomp = True, n_max = n_max,\
                        mid_word = mid_word)



if __name__=='__main__':
    
    Num_of_shapelets_array = [15,21,28,36]
    methods = ['lasso', 'sparse', 'svd', 'lstsq']
    
    ## Range chose so that SNR is in range ~20 -- ~50
    noise_array = np.logspace(1.1, 1.5, 5)
    alpha_ = np.logspace(-5,-1.3,6)
    basis_array = ['Elliptical', 'Polar', 'XY', 'Compound']

    # Generate noisy images
    # galsim images are 78 x 78
    size_X = 78; size_Y=78
    noisy_mat_num = 10

    noise_matrix = np.zeros((size_X*size_Y , noisy_mat_num))
    for i in xrange(noisy_mat_num):
        noise_matrix[:,i] = np.random.randn(size_X*size_Y)

    for noise_scale in noise_array:
        # Select a method for fitting the coefficients
        for basis in basis_array:
            
            for solver in ['svd']:#range(len(methods)): 

                test_stability(solver, basis, \
                        noise_scale, noise_img = noise_matrix, noise_img_num = noisy_mat_num,\
                        size_X = size_X, size_Y = size_Y,\
                        alpha_ = alpha_, Num_of_shapelets_array = Num_of_shapelets_array)
