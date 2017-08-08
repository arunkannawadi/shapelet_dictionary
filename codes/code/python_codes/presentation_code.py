"""
Get some images for the presentation ready

Show the easy manipulation with shapelets:
------------------------------------------
1. Make a dictionary - rotate it and get a rotated galasxy
2. Scaling of the flux is easy
3. Show the KSB and FAM comparison when the probelm is solved
"""

import sys,os
import numpy as np
import matplotlib.pyplot as plt

from scipy.special import hermitenorm
from scipy.integrate import quad
from pylab import imshow
from mpl_toolkits.axes_grid1 import make_axes_locatable

## Import the util package for the weights
from utils.galsim_utils import *
from utils.I_O_utils import *
from utils.shapelet_utils import *

#import pdb; pdb.set_trace()

import pyfits
import galsim
import math


## Custom solver functions
from solver_routines import *
from plotting_routines import *
from shapelet_dicts import *

## Set up global LaTeX parsing
plt.rc('text', usetex=True)
plt.rc('font', **{'family' : "sans-serif"})
params = {'text.latex.preamble' : [r'\usepackage{siunitx}', \
                r'\usepackage[utf8]{inputenc}', r'\usepackage{amsmath}']}
plt.rcParams.update(params)

def _get_shear(img_):

    galsim_img = galsim.Image(img_, scale = 1.0)
    shear_data = galsim_img.FindAdaptiveMom()

    return shear_data.observed_shape.g1, shear_data.observed_shape.g2


if __name__ == '__main__':

    ## Get the image from the galsim real galaxy catalog

    cube_real = pyfits.getdata('../../data/cube_real.fits')
    cube_noiseless = pyfits.getdata('../../data/cube_real_noiseless.fits')

    basis = 'XY_Elliptical'; N1 = 20; N2=20; solver = 'svd'; n_max=55; 
    Num_of_shapelets = 36; alpha = 0.1; 
    selected_imgs = [0, 25, 51, 75, 94];
    mid_word = str(\
                sum_max_order(basis,get_max_order(basis,Num_of_shapelets))) \
                + '_' + basis


    ## Background is the same for all images from the
    ## galsim example catalogue
    background = 1.e6*0.16**2
    

    Path_noisy_0 = '/home/kostic/Documents/_Presentation/Observed/'
    Path_noiseless_0 = '/home/kostic/Documents/_Presentation/Noiseless/'
    Path_theta_noisy_0 = Path_noisy_0 + 'theta_rotation/'
    Path_theta_noiseless_0 = Path_noiseless_0 + 'theta_rotation/'
    Path_scale_noisy_0 = Path_noisy_0 + 'scaled/'
    Path_scale_noiseless_0 = Path_noiseless_0 + 'scaled/'

    for idx in selected_imgs:
        str_img_idx = str("%d" % (idx))
        
        Path_noisy = Path_noisy_0 + str_img_idx + '/'
        Path_noiseless = Path_noiseless_0 + str_img_idx + '/'
        Path_theta_noisy = Path_theta_noisy_0 + str_img_idx + '/'
        Path_theta_noiseless = Path_theta_noiseless_0 + str_img_idx + '/'
        Path_scale_noisy = Path_scale_noisy_0 + str_img_idx + '/'
        Path_scale_noiseless = Path_scale_noiseless_0 + str_img_idx + '/'

        img_real = cube_real[idx] - background
        img_noiseless = cube_noiseless[idx] - background
        
        gal_real = galsim.Image(img_real, scale = 1.0)
        gal_noiseless = galsim.Image(img_noiseless, scale = 1.0)

        ## Get the real beta scales and rotation
        shape_real = gal_real.FindAdaptiveMom()
        shape_noiseless = gal_real.FindAdaptiveMom()

        image_data_real = get_moments(shape_real)
        image_data_noiseless  = get_moments(shape_noiseless)
        
        ## Get the correct beta scale here
        if 'Compound' in basis:
            sigma_real = image_data_real[2]
            sigma_noiseless = image_data_noiseless[2]
            beta_array_real = \
                    [\
                    sigma_real/4., sigma_real/2., sigma_real, \
                    sigma_real*2, sigma_real*4]
            beta_array_noiseless = \
                    [\
                    sigma_noiseless/4.,sigma_noiseless/2., sigma_noiseless, \
                    sigma_noiseless*2, sigma_noiseless*4]
        else:
            beta_array_real = [image_data_real[2]]
            beta_array_noiseless = [image_data_noiseless[2]]

        print "Getting the observed image %d data\n" % idx

        ## Get the data for the noisy image
        D_real, reconst_real, coeffs_real, label_arr_real = \
                shapelet_decomposition(\
                    image_data_real,\
                    f_path = Path_noisy, \
                    basis = basis, solver = solver,\
                    image = img_real, \
                    alpha_ = alpha, Num_of_shapelets = Num_of_shapelets,\
                    N1=N1, N2=N2,\
                    make_labels = True, test_basis=True,\
                    plot_decomp = True, plot_sol = True,\
                    n_max = n_max,\
                    beta_array = beta_array_real)
 
        size_X, size_Y = img_real.shape

        ## Rotate the dictionary
        theta_low= np.pi/3.; theta_high = 2*np.pi/3.; num = 3;
        theta_arr = np.linspace(theta_low, theta_high, num=num)
        theta_list = [0.] + list(theta_arr)
        reconst_rotated_arr = []
        reconst_rotated_arr.append(reconst_real)
    
        for theta in theta_arr:
            print "Obtaining rotated dictionary by %f" % theta
            
            image_data_real_rotated = np.asarray(image_data_real).copy(); 
            image_data_real_rotated[3] = image_data_real_rotated[3] + theta 
            
            D_rotated_real, foo, coeffs_rotated_real, foo = \
                    shapelet_decomposition(\
                        image_data_real_rotated,\
                        f_path = Path_noisy, \
                        basis = basis, solver = solver,\
                        image = img_real, \
                        alpha_ = alpha, Num_of_shapelets = Num_of_shapelets,\
                        N1=N1, N2=N2,\
                        make_labels = True, test_basis=True,\
                        plot_decomp = False, plot_sol = False,\
                        n_max = n_max,\
                        beta_array = beta_array_real)

            reconst_rotated_arr.append(np.dot(D_rotated_real, coeffs_real).reshape(size_X,size_Y)) 
        
        nrows = int(np.round((num+1)/2.))
        ncols = nrows
        fig, ax = plt.subplots(nrows = nrows, ncols = ncols, figsize = (10,10))
        
        colormap = plt.get_cmap('jet')
        vmin, vmax = reconst_real.min(), reconst_real.max() 
        
        ## Get the range for the colorbar
        norm = mpl.colors.Normalize(vmin=vmin,vmax=vmax)
        ## create a ScalarMappable and initialize a data structure
        s_m = cm.ScalarMappable(cmap=colormap, norm=norm); s_m.set_array([])
        
        t=0
        for i in xrange(nrows):
            for j in xrange(ncols):
                theta_ = theta_list[t]
                str_theta = str("%.1f" % (np.degrees(theta_)))
                ax_ = ax[i,j]
                im = \
                    ax_.imshow(reconst_rotated_arr[t], vmin = vmin, vmax=vmax)
                ax_.set_title('Rotation for \n' \
                        + r'$\displaystyle \theta = $'\
                        + str_theta + r'$\displaystyle ^{o}$')
                t+=1
                ## Locate the axes for colorbar
                divider = make_axes_locatable(ax_)
                cax = divider.append_axes("right", size="5%")
                fig.colorbar(im, format = '%.2e', cax = cax)
        

        mkdir_p(Path_theta_noisy)
        plt.tight_layout()
        plt.savefig(Path_theta_noisy \
                + solver + '_' + mid_word +'_' + str_img_idx + '_.png')
        plt.close()

        print "Getting the noiseless image %d data\n" % idx
        
        ## Get the data for the noiselss images
        D_noiseless, reconst_noiseless, coeffs_noiseless, label_arr_noiseless = \
                shapelet_decomposition(\
                    image_data_noiseless,\
                    f_path = Path_noiseless, \
                    basis = basis, solver = solver,\
                    image = img_noiseless, \
                    alpha_ = alpha, Num_of_shapelets = Num_of_shapelets,\
                    N1=N1, N2=N2,\
                    make_labels = True, test_basis=True,\
                    plot_decomp = False, plot_sol = True,\
                    flag_gaussian_fit = False,\
                    n_max = n_max,\
                    beta_array = beta_array_noiseless)
        
        ## Rotate the dictionary
        reconst_rotated_arr = []
        reconst_rotated_arr.append(reconst_noiseless)
    
        for theta in theta_arr:
            print "Obtaining rotated dictionary by %f" % theta
            
            image_data_real_rotated = np.asarray(image_data_noiseless).copy(); 
            image_data_real_rotated[3] = image_data_real_rotated[3] + theta 
            
            D_rotated_noiseless, foo, coeffs_rotated_noiseless, foo = \
                    shapelet_decomposition(\
                        image_data_real_rotated,\
                        f_path = Path_noiseless, \
                        basis = basis, solver = solver,\
                        image = img_noiseless, \
                        alpha_ = alpha, Num_of_shapelets = Num_of_shapelets,\
                        N1=N1, N2=N2,\
                        make_labels = True, test_basis=True,\
                        plot_decomp = False, plot_sol = False,\
                        n_max = n_max,\
                        beta_array = beta_array_real)

            reconst_rotated_arr.append(\
                    np.dot(D_rotated_noiseless, coeffs_noiseless).reshape(size_X,size_Y)) 
        
        nrows = int(np.round((num+1)/2.))
        ncols = nrows
        fig, ax = plt.subplots(nrows = nrows, ncols = ncols, figsize=(10,10))
        
        colormap = plt.get_cmap('jet')
        vmin, vmax = reconst_real.min(), reconst_real.max() 
        
        ## Get the range for the colorbar
        norm = mpl.colors.Normalize(vmin=vmin,vmax=vmax)
        ## create a ScalarMappable and initialize a data structure
        s_m = cm.ScalarMappable(cmap=colormap, norm=norm); s_m.set_array([])
        
        t=0
        for i in xrange(nrows):
            for j in xrange(ncols):
                theta_ = theta_list[t]
                str_theta = str("%.1f" % (np.degrees(theta_)))
                ax_ = ax[i,j]
                im = \
                    ax_.imshow(reconst_rotated_arr[t], vmin = vmin, vmax=vmax)
                ax_.set_title('Rotation for \n' \
                        + r'$\displaystyle \theta = $'\
                        + str_theta + r'$\displaystyle ^{o}$')
                t+=1
                ## Locate the axes for colorbar
                divider = make_axes_locatable(ax_)
                cax = divider.append_axes("right", size="5%")
                fig.colorbar(im, format = '%.2e', cax = cax)
       
        mkdir_p(Path_theta_noiseless)
        plt.tight_layout()
        plt.savefig(Path_theta_noiseless \
                + solver + '_'+mid_word+'_' + str_img_idx + '_.png')
        plt.close()        
        
        ## Scale the coefficients and show
        num = 3
        low = 1.6; high = 2;
        scales = np.linspace(low,high, num = num)
        scales_ = [1.] + list(scales)
        reconst_scaled_real = [reconst_real]
        beta_arr = np.asarray(beta_array_real)

        for scale in scales:
            beta_arr_ = scale * beta_arr
            image_data_ = np.asarray(image_data_real).copy()
            if not('Compound' in basis):
                image_data_[2] = image_data_[2]*scale

            D_scaled_real, foo, coeffs_scaled_real, foo = \
                    shapelet_decomposition(\
                        image_data_,\
                        f_path = Path_noisy, \
                        basis = basis, solver = solver,\
                        image = img_real, \
                        alpha_ = alpha, Num_of_shapelets = Num_of_shapelets,\
                        N1=N1, N2=N2,\
                        make_labels = True, test_basis=True,\
                        plot_decomp = False, plot_sol = False,\
                        n_max = n_max,\
                        beta_array = beta_arr_)
            reconst_scaled_real.append(np.dot(D_scaled_real,coeffs_real).reshape(size_X,size_Y))

        nrows = int(np.round((num+1)/2.))
        ncols = nrows
        fig, ax = plt.subplots(nrows = nrows, ncols = ncols, figsize=(10,10))
        
        colormap = plt.get_cmap('jet') 
        
        t=0
        for i in xrange(nrows):
            for j in xrange(ncols):
                str_scale = str("%.1f" % (scales_[t]))
                ax_ = ax[i,j]
                img_ = reconst_scaled_real[t]
                g1,g2 = _get_shear(img_)
                str_g1 = str("%.4f" % (g1)); str_g2 = str("%.4f" % (g2))
                im = \
                    ax_.imshow(img_, vmin = img_.min(), vmax=img_.max())
                ax_.set_title('Scaled for \n' \
                        + r'$\displaystyle \eta = $'\
                        + str_scale\
                        + '\n'\
                        + r'$\displaystyle g_1 = $'\
                        + str_g1 + '\t'\
                        + r'$\displaystyle g_2 = $'\
                        + str_g2)
                t+=1
                ## Locate the axes for colorbar
                divider = make_axes_locatable(ax_)
                cax = divider.append_axes("right", size="5%")
                fig.colorbar(im, format = '%.2e', cax = cax)
       
        mkdir_p(Path_scale_noisy)
        try:
            plt.tight_layout()
        except RuntimeError as error:
            print "RuntimeError {0}".format(error)
        plt.savefig(Path_scale_noisy \
                + solver + '_'+mid_word+'_' + str_img_idx + '_.png')
        plt.close()

        ## Noiseless
        reconst_scaled_noiseless = [reconst_noiseless]
        beta_arr_noiseless = np.asarray(beta_array_noiseless)

        for scale in scales:
            beta_arr_ = scale * beta_arr_noiseless
            image_data_ = np.asarray(image_data_noiseless).copy()
            if not('Compound' in basis):
                image_data_[2] = image_data_[2]*scale
            D_scaled_noiseless, foo, coeffs_scaled_noiseless, foo = \
                    shapelet_decomposition(\
                        image_data_,\
                        f_path = Path_noisy, \
                        basis = basis, solver = solver,\
                        image = img_real, \
                        alpha_ = alpha, Num_of_shapelets = Num_of_shapelets,\
                        N1=N1, N2=N2,\
                        make_labels = True, test_basis=True,\
                        plot_decomp = False, plot_sol = False,\
                        n_max = n_max,\
                        beta_array = beta_arr_)
            reconst_scaled_noiseless.append(np.dot(D_scaled_noiseless,coeffs_noiseless).reshape(size_X,size_Y)) 
        
        nrows = int(np.round((num+1)/2.))
        ncols = nrows
        fig, ax = plt.subplots(nrows = nrows, ncols = ncols, figsize = (10,10))
        
        colormap = plt.get_cmap('jet') 

        t=0
        for i in xrange(nrows):
            for j in xrange(ncols):
                str_scale = str("%.1f" % (scales_[t]))
                ax_ = ax[i,j]
                img_ = reconst_scaled_noiseless[t]
                g1,g2 = _get_shear(img_)
                str_g1 = str("%.4f" % (g1)); str_g2 = str("%.4f" % (g2))
                im = \
                    ax_.imshow(img_, vmin = img_.min(), vmax=img_.max())
                ax_.set_title('Scaled for \n' \
                        + r'$\displaystyle \eta = $'\
                        + str_scale\
                        + '\n'\
                        + r'$\displaystyle g_1 = $'\
                        + str_g1 + '\t'\
                        + r'$\displaystyle g_2 = $'\
                        + str_g2)
                t+=1
                ## Locate the axes for colorbar
                divider = make_axes_locatable(ax_)
                cax = divider.append_axes("right", size="5%")
                fig.colorbar(im, format = '%.2e', cax = cax)
       
        mkdir_p(Path_scale_noiseless)
        try:
            plt.tight_layout()
        except RuntimeError as error:
            print "RuntimeError {0}".format(error)
            pass

        plt.savefig(Path_scale_noiseless \
                + solver + '_'+mid_word+'_' + str_img_idx + '_.png')
        plt.close()
