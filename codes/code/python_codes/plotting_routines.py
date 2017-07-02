import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from utils.I_O_utils import *
from utils.shapelet_utils import *

## Set up global LaTeX parsing
plt.rc('text', usetex=True)
plt.rc('font', **{'family' : "sans-serif"})
params = {'text.latex.preamble' : [r'\usepackage{siunitx}', \
                r'\usepackage[utf8]{inputenc}', r'\usepackage{amsmath}']}
plt.rcParams.update(params)

#import pdb; pdb.set_trace()

def coeff_plot2d(coeffs,N1,N2,\
        ax=None,fig=None,\
        orientation='vertical',\
        f_coef_output = ''):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import LogNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    ## Double check write out to file
    ## coeffs

    if f_coef_output != '':
        f = open(f_coef_output, "w")
        f.write("N_1\tN_2\tCoeff values\n")

    if ax is None:
      fig, ax = plt.subplots()

    coeffs_reshaped = None

    if (coeffs.shape != (N1,N2)):
        coeffs_reshaped = coeffs.reshape(N1,N2) 
    else:
        coeffs_reshaped = coeffs
    
    ## Output to the file only nonzero values of coefs
    if f_coef_output != '':
        idx_nonzero = np.where(coeffs_reshaped != 0)
        for n,m in zip(idx_nonzero[0], idx_nonzero[1]):
            f.write("%d\t%d\t%.3f\n" % (n,m,coeffs_reshaped[n,m]))
        f.close()

    #coeffs_reshaped /= coeffs_reshaped.max()
    im = ax.imshow(coeffs_reshaped,cmap=cm.bwr,interpolation='none')
    
    ## Force colorbars besides the axes
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size='5%')    

    fig.colorbar(im,cax=cax,orientation=orientation)
    return fig,ax

def coeff_plot_polar(coeffs, N1,N2, \
        len_beta = 1, N_range = 10,ax = None, fig = None, colormap = cm.bwr,\
        orientation = 'vertical',\
        f_coef_output = ''):
    """
    Plot the values of coeffs in the triangular grid
    """
    import matplotlib as mpl
    import matplotlib.cm as cm
    from matplotlib.patches import Rectangle
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    ## For double check / if color scale is not too good
    ## write out the values of 
    ## the coefs
    if f_coef_output != '':
        f = open(f_coef_output, "w")
        f.write("N\tM\tCoeff value\n")
    ## Initialize the array for the colors / intenzities of the coefs
    color_vals = []

    ## For now 15 is the selected N_range
    x = []
    y = []
    
    ## Check if order of n in basis (N1 currently) is smaller than N_range
    if N1 < N_range:
        N_range = N1
    
    k = 0
    for n in xrange(N_range):
        for m in xrange(-n,n+1,2):
            x.append(n)
            color_vals.append(coeffs[k])
            if coeffs[k] != 0:
                f.write("%d\t%d\t%.3f\n" % (n,m, coeffs[k]))
            k += 1
        ## Make appropriate y coord
        ## So that the squares don't overlap
        ## and there is no white space between
        y.append(np.linspace(-n/2.,n/2.,n+1))

    f.close()
    ## Merge everything into one array of the same shape as x
    x = np.asarray(x)
    y = np.concatenate(y[:])    
    color_vals = np.abs(np.asarray(color_vals))
    ## Control the size of squares
    dx = [x[1]-x[0]]*len(x) 
    
    ## Get the range for the colorbar
    norm = mpl.colors.Normalize(
        vmin=np.min(color_vals),
        vmax=np.max(color_vals))

    ## choose a colormap
    c_m = colormap

    ## create a ScalarMappable and initialize a data structure
    s_m = cm.ScalarMappable(cmap=c_m, norm=norm); s_m.set_array([])

    if fig == None:
        fig, ax = plt.subplots()
    
    ax.set_ylabel('m')
    ax.set_xlabel('n')
    ax.set_aspect('equal')
    ax.axis([min(x)-1., max(x)+1., min(y)-1., max(y)+1.])
    
    ## Normalize the values so that it is easier to plot
    color_vals = color_vals / np.max(color_vals)

    ## Add the coeffs as squares
    ## without any white spaces remaining
    for x,y,c,h in zip(x,y,color_vals,dx):
        ax.add_artist(\
                Rectangle(xy=(x-h/2., y-h/2.),\
                linewidth=3, color = colormap(c),\
                width=h, height=h))
    
    ## Locate the axes for colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%")

    fig.colorbar(s_m, cax = cax)
    
    return fig, ax

def plot_decomposition(basis, cube, img_idx, size_X, size_Y, \
        base_coefs,N1,N2,shapelet_reconst_array, signal, residual_array,\
        residual_energy_fraction_array,recovered_energy_fraction_array, Path,\
        beta_array = []):

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
    @param beta_array Array with betas used // Added for the compound basis

    """ 
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    if basis =='Compound':
        base_coefs = base_coefs.reshape(N1,N2*len(beta_array))    
    elif basis == 'XY' or basis == 'Elliptical':
        base_coefs = base_coefs.reshape(N1,N2)

    for i in xrange(len(beta_array)):
       
        shapelet_reconst = shapelet_reconst_array[i]
        residual = residual_array[i]
        residual_energy_fraction = residual_energy_fraction_array[i]
        recovered_energy_fraction = recovered_energy_fraction_array[i]

        str_beta = str("%.3f" % (beta_array[i]))

        if basis != 'Polar':
            left_N2 = 0 + i*N2
            right_N2 = N2 + i*N2
            coefs = base_coefs[:N1, left_N2:right_N2]
        else:
            coefs = base_coefs
            
        if np.count_nonzero(coefs) != 0:

            print 'beta ', beta_array[i], 'Decomp ', coefs.shape

            fig, ax = plt.subplots(2,2, figsize = (10, 10))
            if basis == 'XY' or basis == 'Elliptical' or basis == 'Compound':
                coeff_plot2d(coefs,N1,N2,\
                        ax=ax[1,1],fig=fig,\
                        f_coef_output = Path + '_' + str_beta + '_.txt')
            elif basis == 'Polar':
                coeff_plot_polar(coefs,N1,N2,\
                        ax=ax[1,1], fig=fig,
                        f_coef_output = Path + '_' + str_beta + '_.txt')

            vmin, vmax = min(shapelet_reconst.min(),signal.min()), \
                    max(shapelet_reconst.max(),signal.max())

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
            ax[1,1].grid(lw = 2, which = 'both')
            ax[1,1].set_title('Values of coefficients')
            fig.suptitle('Shapelet Basis decomposition')
            
            fig.tight_layout()
            plt.savefig(Path + '_' + str_beta + '_.png')
            plt.clf()

def plot_solution(basis, N1,N2,image_initial,size_X, size_Y,\
        reconst, residual, coefs_initial,\
        recovered_energy_fraction, residual_energy_fraction, \
        n_nonzero_coefs, noise_scale, Path,\
        beta_array = []):

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
    
    ## Divide the initial matrix to 
    ## submatrices which correspond to the different basis
    
    ## Initialize this for cartesian and compound
    ## but for polar it is not needed
    tmp_coefs_initial = np.zeros((len(beta_array),N1,N2))
    if basis =='Compound':
        for i in xrange(len(beta_array)):
            tmp_coefs_initial[i] = coefs_initial[i*(N1*N2): N1*N2 + i*(N1*N2)].reshape(N1,N2)
    elif basis == 'XY' or basis == 'Elliptical':
            tmp_coefs_initial[i] = coefs_initial.reshape(N1,N2)
    
    for i in xrange(len(beta_array)):
       
        str_beta = str("%.3f" % (beta_array[i]))

        if basis != 'Polar':
            left_N2 = 0 + i*N2
            right_N2 = N2 + i*N2
            coefs = tmp_coefs_initial[i]
        else:
            coefs = coefs_initial
        
        if np.count_nonzero(coefs) != 0:
            print 'beta ', beta_array[i], 'shape: ', coefs.shape


            fig2, ax2 = plt.subplots(2,2, figsize = (10,10))
            vmin, vmax = min(reconst.min(),image_initial.min()), \
                    max(reconst.max(),image_initial.max())
            
            im00 = ax2[0,0].imshow(image_initial, aspect = '1', vmin=vmin, vmax=vmax)
            im01 = ax2[0,1].imshow(\
                    reconst.reshape(size_X,size_Y), aspect = '1', vmin=vmin, vmax=vmax)
            im10 = ax2[1,0].imshow(residual.reshape(size_X,size_Y), aspect = '1')

            if basis == 'XY' or basis == 'Elliptical' or basis == 'Compound':
                coeff_plot2d(coefs,N1,N2,\
                        ax=ax2[1,1],fig=fig2,\
                        f_coef_output = Path + '_' + str_beta + '_.txt') 
            elif basis == 'Polar':
                coeff_plot_polar(coefs,N1,N2,\
                        ax=ax2[1,1],fig=fig2,\
                        f_coef_output = Path + '_' + str_beta + '_.txt')     

            ax2[1,1].grid(lw=2)
            
            # Force the colorbar to be the same size as the axes
            divider00 = make_axes_locatable(ax2[0,0])
            cax00 = divider00.append_axes("right", size="5%")

            divider01 = make_axes_locatable(ax2[0,1])
            cax01 = divider01.append_axes("right", size="5%")

            divider10 = make_axes_locatable(ax2[1,0])
            cax10 = divider10.append_axes("right", size="5%")  
            
            fig2.colorbar(im00,cax=cax00);
            fig2.colorbar(im01,cax=cax01); fig2.colorbar(im10,cax=cax10)
            ax2[0,0].set_title('Original (noisy) image'); 
            ax2[0,1].set_title('Reconstructed image - Frac. of energy = '\
                    +str(np.round(recovered_energy_fraction,4)))
            ax2[1,0].set_title('Residual image - Frac. of energy = '\
                    +str(np.round(residual_energy_fraction,4))); 
            fig2.suptitle('Sparse decomposition from an semi-intelligent Dictionary')

            if (noise_scale == 0):
                ax2[1,1].set_title('Values of coefficients - '\
                        + str(n_nonzero_coefs) \
                        + '\n' \
                        + 'beta - ' + str_beta)
            else:
                ax2[1,1].set_title('Rel. diff in values ' \
                        + r'$\displaystyle \left|\frac{N.C_i}{O.C_i} - 1 \right|$'\
                        + ' - ' + str(n_nonzero_coefs) \
                        + '\n' \
                        + 'beta - ' + str_beta)
            
            fig2.tight_layout()
            plt.savefig(Path + '_' + str_beta + '_.png')
            plt.clf()

def stability_plots(basis,solver,coefs,\
        N1,N2,\
        f_path_to_save,y_axis_scale = '',\
        ax_title = '', f_coef_output = ''):
    
    fig, ax = plt.subplots()
        
    if basis == 'XY' or basis == 'Elliptical' or basis == 'Compound':
        coeff_plot2d(coefs,N1,N2,ax=ax,fig=fig,\
                f_coef_output = f_coef_output) 
    elif basis == 'Polar':
        coeff_plot_polar(coefs,N1,N2,ax=ax,fig=fig,\
                f_coef_output = f_coef_output)
    
    if y_axis_scale != '':
        ax.set_yscale(y_axis_scale)
    ax.grid(lw=2)
    ax.set_aspect('equal')
    ax.set_title(ax_title)

    fig.tight_layout()
    plt.savefig(f_path_to_save + '_.png')
    plt.clf()

def plot_stability(coeff_stability, coeff_0, N1, N2, noise_img_num, \
        n_max = 20, label_arr = None,\
        beta_array = [], signal_to_noise = None, \
        basis = 'Polar', solver = 'lasso', \
        path_to_save = 'Plots/lasso/Stability/',\
        mid_word = ''):
    
    """
    Plot the stability matrices and variance matrices for selected solver method
    """
    ## Initialize the folder for saving the images
    mkdir_p(path_to_save)

    
    ## Initialize the stability and variance arrays
    
    len_coeffs = coeff_stability.shape[0]
    coeff_stability_res = np.zeros(len_coeffs)
    coeff_mean_value = np.zeros(len_coeffs)
    coeff_diff = np.zeros(len_coeffs)
    variance = np.zeros(len_coeffs)
    variance_sqrt = np.zeros(len_coeffs)
    
    ## Find the mean coefficients and the variance
    ## <N.C>_i and Var(N.C_i)
    for i in xrange(len_coeffs):
        
        ## Add all the values in different noise realizations
        ## of the same coordinate
        coeff_stability_res[i] = np.sum(coeff_stability[i,:])
        
        ## Calculate the variance of the i_th coordinate
        std_i = np.std(coeff_stability[i,:])
        variance_sqrt[i] = np.sqrt(std_i)

        ## Calculate the mean of the i_th coordinate
        coeff_stability_res[i] = coeff_stability_res[i]/noise_img_num
        coeff_diff[i] = coeff_stability_res[i] - coeff_0[i]
        coeff_mean_value[i] = coeff_stability_res[i]
        if coeff_stability_res[i] != 0:
            variance[i] = std_i/np.abs(coeff_stability_res[i])
        else:
            variance[i] = 0
    
    ## Asses the difference
    ## |<N.C>_i / O.C_i - 1|
    for i in xrange(len_coeffs):
        if coeff_0[i] != 0:
            coeff_stability_res[i] = np.abs(coeff_stability_res[i]/coeff_0[i] - 1)
        else:
            coeff_stability_res[i] = 0

    ## Add the scatter plot
    ## How many points to plot  
    n_max = sum_max_order(basis, get_max_order(basis,n_max))    
    n_nonzero = np.count_nonzero(coeff_stability_res)
    n_nonzero = sum_max_order(basis,get_max_order(basis, n_nonzero))

    ## Take the biggest possible number of 
    ## coefficients
    if n_max <= n_nonzero:
        N_plot = n_max
    else:
        N_plot = n_nonzero

    ## Indices measured according to the smallest number of
    ## nonzero values in these arrays
    nonzero_var_sqrt = len(np.where(variance_sqrt != 0)[0])
    nonzero_coeff_mean = len(np.where(coeff_mean_value != 0)[0])
    
    ## If you can get all of the nonzero if not
    ## then just N_plot values
    ## If N_plot > nonzero * it returns the whole set of indices
    if nonzero_var_sqrt <= nonzero_coeff_mean:
        get_idx = np.where(variance_sqrt != 0)[0][:N_plot]
    else:
        get_idx = np.where(coeff_mean_value != 0)[0][:N_plot]

    coeff_mean_r = coeff_mean_value[get_idx]
    coeff_res_r = coeff_stability_res[get_idx]
    variance_sqrt_r = variance_sqrt[get_idx] 
    label_arr_r = label_arr[get_idx]

    arange_x = np.arange(len(get_idx))
    
    fig_scat, ax_scat = plt.subplots(2, sharex=True)

    ax_scat[0].set_title(\
            'Scatter plot of '\
            + r'$\displaystyle \left<N.C_i\right>$'\
            + 'for first %d coeffs'\
            % (N_plot))

    ax_scat[1].set_title(\
            'Scatter plot of'\
            + r'$\displaystyle \left|\frac{\left<N.C._i\right>}{O.C._i} - 1 \right|$')
    ax_scat[0].set_yscale('symlog')
    ax_scat[1].set_yscale('symlog')
    ax_scat[0].errorbar(arange_x, coeff_mean_r, yerr=variance_sqrt_r, fmt='bo', \
            label='Coeff. value')
    ax_scat[1].errorbar(arange_x, coeff_res_r, yerr=variance_sqrt_r,fmt='ro',\
            label='Coeff. stability')
    plt.xticks(arange_x, label_arr_r)
    ax_scat[1].set_xticklabels(ax_scat[1].get_xticklabels(),rotation=90, horizontalalignment = 'right')
    
    ax_scat[0].tick_params(axis='x', which ='both', pad = 10)
    ax_scat[1].tick_params(axis='x', which ='both', pad = 10)
    ax_scat[0].set_xlim(min(arange_x) - 1, max(arange_x) + 1)
    ax_scat[1].set_xlim(min(arange_x) - 1, max(arange_x) + 1)

    ## Set the font of the ticks
    for tick in ax_scat[0].xaxis.get_major_ticks():
        tick.label.set_fontsize(7)
    for tick in ax_scat[0].xaxis.get_minor_ticks():
        tick.label.set_fontsize(7)
    
    fig_scat.tight_layout()
    plt.savefig(path_to_save + solver + '_' + mid_word + "_scatter_coefs.png")
    plt.clf()

    ## Calculate the relative variance of decompositions
    ## Coeff_stability is already in the needed form
    ## variance_i = Var(N.C_i)
    
    if basis =='Compound':
        variance_initial = variance.reshape(N1,N2*len(beta_array))
        variance_sqrt_initial = variance_sqrt.reshape(N1,N2*len(beta_array))
        coefs_initial = coeff_stability_res.reshape(N1,N2*len(beta_array))
        coefs_diff_initial = coeff_diff.reshape(N1,N2*len(beta_array))
    elif basis == 'XY' or basis == 'Elliptical':
        variance_sqrt_initial = variance_sqrt.reshape(N1,N2)
        variance_initial = variance.reshape(N1,N2)
        coefs_initial = coeff_stability_res.reshape(N1,N2)
        coefs_diff_initial = coeff_diff.reshape(N1,N2)
    
    
    str_s_to_n = str("%.3e" % (signal_to_noise)) 

    for i in xrange(len(beta_array)):
        
        str_beta = str("%.3f" % (beta_array[i]))

        if basis != 'Polar':
            left_N2 = 0 + i*N2
            right_N2 = N2 + i*N2
            variance = variance_initial[:N1, left_N2:right_N2]
            variance_sqrt = variance_sqrt_initial[:N1, left_N2:right_N2]
            coefs = coefs_initial[:N1,left_N2:right_N2]
            coefs_diff = coefs_diff_initial[:N1,left_N2:right_N2]
        else:
            coefs = coeff_stability_res
            coefs_diff = coeff_diff

        ## Plot the stability of coeffs
        ax_title = 'Diff of coefs '\
            + r'$\displaystyle \left<N.C._i\right> - O.C._i$' \
            + '\n' \
            + 'N.C is averaged over the number of noise realizations' \
            + '\n' \
            + 'S/N = ' + str_s_to_n\
            + '\n'\
            + 'beta - ' + str_beta
        
        f_path_to_save = path_to_save + solver + '_diff_'+mid_word+'_'+str_s_to_n+'_'\
            +str_beta

        stability_plots(basis,solver,coefs_diff,\
                N1,N2,\
                f_path_to_save,\
                ax_title = ax_title, f_coef_output = f_path_to_save + '_.txt')

        ## Plot the stability of coeffs
        ax_title = 'Stability of coefs '\
            + r'$\displaystyle \left|\frac{<N.C.>_i}{O.C._i} - 1\right|$' \
            + '\n' \
            + 'N.C is averaged over the number of noise realizations' \
            + '\n' \
            + 'S/N = ' + str_s_to_n\
            + '\n'\
            + 'beta - ' + str_beta
        
        f_path_to_save = path_to_save + solver + '_stability_'+mid_word+'_'+str_s_to_n+'_'\
            +str_beta

        stability_plots(basis,solver,coefs,\
                N1,N2,\
                f_path_to_save,\
                ax_title = ax_title, f_coef_output = f_path_to_save + '_.txt')


        ## Plot relative variance
        ax_title = 'Rel. variance matrix '\
                + r'$\displaystyle \sigma\left(N.C._i\right) / |\left< N.C._i\right>|$' \
                + '\n' \
                + 'S/N = ' + str_s_to_n\
                + '\n'\
                + 'beta - ' + str_beta
        
        f_path_to_save = path_to_save + solver + '_variance_rel_'+mid_word+'_'+str_s_to_n+'_'\
            +str_beta

        stability_plots(basis,solver,variance,\
                N1,N2,\
                f_path_to_save,\
                ax_title = ax_title, f_coef_output = f_path_to_save + '_.txt')

        ## Plot standard deviation

        ax_title = 'Variance matrix '\
                + r'$\displaystyle \sigma\left(N.C._i\right)$' \
                + '\n' \
                + 'S/N = ' + str_s_to_n\
                + '\n'\
                + 'beta - ' + str_beta
        
        f_path_to_save = path_to_save + solver + '_variance_sqrt_' + mid_word + '_' + str_s_to_n\
                +str_beta
            
        stability_plots(basis,solver,variance_sqrt,\
                N1,N2,\
                f_path_to_save,\
                ax_title = ax_title, f_coef_output = f_path_to_save + '_.txt')
