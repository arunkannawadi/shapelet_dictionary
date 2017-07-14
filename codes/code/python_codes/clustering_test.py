import galsim
import pyfits
import time
import numpy as np
import matplotlib.pyplot as plt

from shapelet_dicts import *
from plotting_routines import plot_stability

## Import the util package for the weights
from utils.galsim_utils import *
from utils.I_O_utils import *
from utils.shapelet_utils import *

import pdb; pdb.set_trace()

def _gen_cluster_data(\
        basis, solver,\
        N1=20,N2 = 20,\
        beta_array = [1.881,2.097, 2.531, 3.182, 4.918]):
    
    start_time= time.time()

    cube = pyfits.getdata("../../data/cube_real.fits")
    background = 1e6*0.16**2
    image_data = np.zeros(5)

    n_max = 55; Num_of_shapelets = 21;
    f_path = 'Plots/Cluster_test/'
  
    k = 0
    coeffs_val_cluster = []; 
    label_arr_cluster = []

    for image in cube[:20]:
        image = image - background
        galsim_img = galsim.Image(image, scale = 1.0)
        shape_data = galsim_img.FindAdaptiveMom()

        print "image flux: ", shape_data.moments_amp
        image /= shape_data.moments_amp

        x0,y0,sigma,theta,q = get_moments(shape_data)
    
        print "Image %d data" % (k)
        print "x0\ty0\tsigma\ttheta\tq\n"
        print "%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n" % (x0,y0,sigma,theta,q)

        image_data[0] = x0; image_data[1] = y0; image_data[2] = sigma;
        image_data[3] = theta; image_data[4] = q

        Path = f_path + str("%d" % (k)) + '/'; k+=1
        mkdir_p(Path)

        reconst, coeffs, label_arr = shapelet_decomposition(\
                image_data,\
                f_path = Path, \
                basis = basis, solver = solver,\
                image = image, Num_of_shapelets = Num_of_shapelets,\
                N1=N1, N2=N2,\
                make_labels = True, n_max = n_max,\
                beta_array = beta_array)
        coeffs_val_cluster.append(coeffs)
        label_arr_cluster.append(label_arr)

    print("Elapsed %s s" % (time.time() - start_time))
    return np.asarray(coeffs_val_cluster), np.asarray(label_arr_cluster)

def _visualize(\
        coeffs_val_cluster, label_arr_cluster,\
        basis,solver,\
        beta_array,\
        N1=20,N2=20):

    """
    Bin the coefficient values and plot the result. Account for only nonzero values in the binning.

    Parameters:
    -----------
    coeffs_val_cluster : values of the coefficients to be binned
    label_arr_cluster : labels of the shapelets for the binning
    basis : basis in which the decomposition was done
    solver : solver used for fitting the coefficients
    beta_array : array of beta values used for different basis
    
    Optional:
    ---------
    N1,N2 : dimension of the shapelet basis is N1*N2
    """

    ## Number of galaxies included in the statistics
    galaxy_num = coeffs_val_cluster.shape[0]

    ## Every coeffs array is going to have same label indexation
    f_path = 'Plots/Cluster_test/Beta_Scales/' + solver + '/'

    step = 0
    for beta in beta_array:
        
        str_beta = str("%.3f" % (beta))
        
        mkdir_p(f_path + str_beta +'/')
        
        left = step*(N1*N2)
        right = (step+1)*(N1*N2)
        coeffs_val_curr = coeffs_val_cluster[:, left:right]

        indices = np.where(coeffs_val_curr!=0); step+=1;
    
        ## Get the integer numbers representing shapelet labels
        ## for corresponding coefficient values
        idx_1 = indices[0]
        idx_2 = indices[1] 

        ## Get number of distinct shapelets
        ## along with the basis index, to know from which basis
        ## they come from
        sh_labels = []
        basis_labels = []
        for i in xrange(len(idx_2)):
            sh_label = idx_2[i]
            if not(sh_label in sh_labels):
                sh_labels.append(sh_label)
                basis_labels.append(idx_1[i])

        sh_labels = np.asarray(sh_labels)
     
        print "idx_1\n", idx_1
        print "idx_2\n", idx_2
        print "sh_labels\n", sh_labels
        print "basis_labels\n", basis_labels   
        
        for i in xrange(len(sh_labels)):
            sh_label = sh_labels[i]
            basis_label = basis_labels[i]

            fig, ax = plt.subplots()

            ax.set_title('Shapelet ' + label_arr_cluster[basis_label][sh_label])
            
            data_hist = coeffs_val_curr[:, sh_label]
            
            ## Count into the bins only nonzero values
            weights = np.zeros_like(data_hist)
            weights[np.where(data_hist != 0)] = 1

            H, bins, patches = ax.hist(data_hist, \
                    weights = weights, histtype = 'step', align = 'mid')  
            
            ## Set up y labels
            ax.set_yticks(np.linspace(0., galaxy_num, num = 11))
            y_tick_labels = [val/galaxy_num for val in ax.get_yticks()]
            ax.set_ylabel('Fraction of galaxies')
            ax.set_yticklabels(y_tick_labels)

            ## Set up x labels
            few_biggest = 2
            idx_few_biggest = H.argsort()[-few_biggest:]
            for idx in idx_few_biggest:
                height = H[idx]
                if height != 0:
                    x_coord = 0.5*(bins[idx] + bins[idx+1])
                    y_coord = height + 0.05*galaxy_num
                    ax.text(x_coord, y_coord, str("%.2e" % (x_coord)), \
                        fontsize = 'small',ha = 'center', va='bottom')
            
            ax.ticklabel_format(stype='sci', axis = 'x', scilimits = (0,0)) 
            ax.set_xlabel('Coefficient values')
            
            file_save_path = f_path + str_beta+'/' \
                + label_arr_cluster[basis_label][sh_label]
            
            ## Save file with all the bin values
            ## just for check
            f = open(file_save_path + '.txt','w')
            f.write("Bin value\tBin center value\n")
            nonzero_bins = np.where(H != 0)[0]
            for i in nonzero_bins:
                f.write("%.1f\t%.5f\n" % (float(H[i])/galaxy_num, 0.5*(bins[i] + bins[i+1])))
            f.close()

            plt.tight_layout()
            plt.savefig(file_save_path + '.png')
            plt.cla()
            plt.clf()
            plt.close()


if __name__ == "__main__":
    
    ## Generate data_set for clustering
    basis = 'Compound_XY'; solver = 'omp'; 
    beta_array = [1.881, 2.097, 2.531, 3.182, 4.918]
    
    coeffs_val_cluster,label_arr_cluster = \
            _gen_cluster_data(basis, solver, beta_array = beta_array)
    
    coeffs_val_cluster = np.asarray(coeffs_val_cluster)
    label_arr_cluster = np.asarray(label_arr_cluster)

    _visualize(coeffs_val_cluster, label_arr_cluster,basis, solver,beta_array)

