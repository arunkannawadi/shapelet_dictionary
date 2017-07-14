import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from utils.I_O_utils import *

import pdb; pdb.set_trace()

def mkdir_p(mypath):
    """
    Creates a directory. equivalent to using mkdir -p on the command line
    
    @param mypath - local path 
    
    --------------
    Change the root_path in accordance to your wished root path
    """

    from errno import EEXIST
    from os import makedirs,path
    import os

    root_path = '/home/kostic/Documents/codes/code/python_codes/'
    
    try:
        makedirs(root_path + mypath)
    except OSError as exc:
        if exc.errno == EEXIST and path.isdir(root_path + mypath):
            pass
        else: raise


## ## Make a gaussian smothing kernel
#       # xy = np.vstack([n_m,vals])
#       # print xy
#       # ker = gaussian_kde(xy)(xy)         
#       # 
#       # print n_m, n_m.shape
#       # print vals, vals.shape
#       # print ker, ker.shape
#
#       # idx = ker.argsort()
#       # n_m, vals, ker = n_m[idx], vals[idx], ker[idx]
#        
#        fig, ax = plt.subplots()
#    
#       # ## create a ScalarMappable and initialize a data structure
#       # #s_m = cm.ScalarMappable(cmap=ker); s_m.set_array([])
#
#       # ax.scatter(n_m, vals, c=ker, s=50, edgecolor='')    
#       # ### Locate the axes for colorbar
#       # #divider = make_axes_locatable(ax)
#       # #cax = divider.append_axes("right", size="5%")
#       # #
#       # #fig.colorbar(s_m, cax = cax)
#       # #colorbar.set_ticks(np.linspace(counts.min(), counts.max(), counts.max()+1))
#
#        ## Make bin edges centered around each shapelet idx
#        sh_edges = []
#        step = 0
#        for i in xrange(len(n_m)):
#            left_edge = n_m[i] - 10
#            right_edge = n_m[i] + 10
#            if not((left_edge in sh_edges)) and not(right_edge in sh_edges):
#                    sh_edges.append(step + left_edge)
#                    sh_edges.append(step + right_edge)
#            step = right_edge
#
#        sh_edges = np.asarray(sh_edges)
#        sh_edges.sort()
#
#        ## Make edges for the values of coeffs
#        val_edges = []
#        for i in xrange(len(vals)):
#            left_edge = vals[i]-100
#            right_edge = vals[i] + 100
#            if not(left_edge in val_edges) and not(right_edge in val_edges):
#                val_edges.append(left_edge)
#                val_edges.append(right_edge)
#
#        val_edges = np.asarray(val_edges)
#        val_edges.sort()
#        
#        map_ticks_n_m = np.arange(len(sh_edges)/2)
#        map_ticks_vals = np.arange(len(val_edges)/2)
#
#        #hist2d, sh_edges1, val_edges1 = np.histogram2d(n_m,vals)
#        counts, xedges,yedges, img = plt.hist2d(n_m,vals, \
#                bins = (sh_edges, val_edges))
#
#        colorbar = plt.colorbar()
#        colorbar.set_ticks(np.linspace(counts.min(), counts.max(), counts.max()+1))
#
#        plt.xlabel('Shapelets')
#        plt.ylabel('Values of coeffs')
                


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

if __name__ == '__main__':

    coeffs_val_cluster = np.loadtxt('test_cluster_coeff.txt', unpack = True)
    label_arr_cluster = np.loadtxt('test_cluster_label.txt', dtype = 'S10', unpack = True)

    print coeffs_val_cluster
    print label_arr_cluster
