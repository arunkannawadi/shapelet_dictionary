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

#import pdb; pdb.set_trace()

def _gen_cluster_data(\
        basis, solver,\
        N1=20,N2 = 20,\
        beta_array = [1.881,2.097, 2.531, 3.182, 4.918],
        n_max = 55, Num_of_shapelets = 21):

    """
    Decompose images from the given set of galaxy images and return the obtained 
    shapelet labels and shapelet coefficients

    Parameters:
    -----------
    basis       : Basis in which decomposition is going to be made
    solver      : Solver used for finding the coeffs
    N1, N2      : Determining the max size of the basis (20 x 20)
    beta_array  : Array of beta values to be used in the Compound_* basis
    """
    start_time= time.time()

    cube = pyfits.getdata("../../data/cube_real.fits")
    background = 1e6*0.16**2
    image_data = np.zeros(5)

    root_path = '/home/kostic/Documents/codes/code/python_codes/'
    f_path = root_path + 'Plots/Cluster_test/'
  
    
    coeffs_val_cluster = []; 
    label_arr_cluster = []
    flag_fail = 0
    k = 0

    for image in cube:
        image = image - background
        galsim_img = galsim.Image(image, scale = 1.0)
        k+=1

        try:
            shape_data = galsim_img.FindAdaptiveMom() #strict = False, watch out for failure, try block
            flag_moments = 1
        except RuntimeError as error:
            flag_moments = 0
            print("RuntimError: {0}".format(error))
            pass

        print "\n"
        print "Image %d" % (k)
        print "Flag %d" % (flag_moments)
        print "\n"

        if flag_moments == 1:
            print "image flux: ", shape_data.moments_amp
            
            image /= shape_data.moments_amp

            x0,y0,sigma,theta,q = get_moments(shape_data)
        
            print "Image %d data" % (k)
            print "x0\ty0\tsigma\ttheta\tq\n"
            print "%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n" % (x0,y0,sigma,theta,q)

            image_data[0] = x0; image_data[1] = y0; image_data[2] = sigma;
            image_data[3] = theta; image_data[4] = q

            Path = f_path + str("%d" % (k)) + '/';
            mkdir_p(Path)

            try:
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
            except RuntimeError as error:
                f = open('data_backup.txt', 'w')
                f.write("Shapelet label\tCoeff value\n")
                for i in xrange(coeffs_val_cluster.shape[0]):
                    for j in xrange(coeffs_val_cluster.shape[1]):
                        f.write("%s\t%.10f\n" % \
                                (label_arr_cluster[i][j], coeffs_val_cluster[i][j]))
                f.close()
                flag_fail = 1
                print("RuntimeError: {0}".format(error))
                pass
            
    nonzero_coefs = np.count_nonzero(coeffs)
    name_word = basis + '_' + str("%d" % (nonzero_coefs))
    print("Elapsed %s s" % (time.time() - start_time))
    return np.asarray(coeffs_val_cluster), np.asarray(label_arr_cluster), name_word, flag_fail

def _visualize(\
        coeffs_val_cluster, label_arr_cluster,\
        basis,solver,\
        beta_array,\
        N1=20,N2=20,\
        name_word = ''):

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
    root_path  = '/home/kostic/Documents/codes/code/python_codes/'
    f_path = root_path + 'Plots/Cluster_test/Beta_Scales/' + solver + '_' + name_word + '/'

    step = 0
    for beta in beta_array:

        print "Statistics for beta %f" % (beta)

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
        
        for i in xrange(len(sh_labels)):
            sh_label = sh_labels[i]
            basis_label = basis_labels[i]

            fig, ax = plt.subplots()

            ax.set_title('Shapelet ' + label_arr_cluster[basis_label][sh_label])
            
            data_hist = coeffs_val_curr[:, sh_label]
            
            ## Assign weights to zero valued coeffs to 0
            weights = np.zeros_like(data_hist)
            nonzero_picks = np.count_nonzero(data_hist)
            weights[np.where(data_hist != 0)] = 1

            H, bins, patches = ax.hist(data_hist, \
                    weights = weights, \
                    label='Nonzero picks ' + str("%.2f" % (float(nonzero_picks) / galaxy_num )),\
                    histtype = 'step', align = 'mid')  
            
            ## Set up y labels
            ax.set_yticks(np.linspace(0., galaxy_num, num = 11))
            y_tick_labels = [val/galaxy_num for val in ax.get_yticks()]
            ax.set_ylabel('Fraction of galaxies')
            ax.set_yticklabels(y_tick_labels)

            ## Set up x labels and add text of coeff values to plot
            ## where the 2 biggest bins are
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

            plt.legend(loc='best')
            plt.tight_layout()
            plt.savefig(file_save_path + '.png')
            plt.cla()
            plt.clf()
            plt.close()


if __name__ == "__main__":
    
    ## Generate data_set for clustering
    basis = 'Compound_Polar'; solver = 'omp'; 
    n_max = 55; Num_of_shapelets = 21;
    beta_array = [1.881, 2.097, 2.531, 3.182, 4.918]
    
    coeffs_val_cluster,label_arr_cluster,name_word,flag_fail = \
            _gen_cluster_data(\
            basis, solver, \
            n_max = n_max, Num_of_shapelets = Num_of_shapelets, beta_array = beta_array) 
    _visualize(coeffs_val_cluster, label_arr_cluster,basis, solver,beta_array, name_word = name_word)

