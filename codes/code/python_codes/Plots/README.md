Image names are formated as following:  

1st part of the string is the solver included:  
    -- lstsq_* - least squares solution  
    -- lasso_* - lasso regularization method  
    -- SVD_* - singular value decomposition  
    -- Sparse_* - Orthogonal matching pursuit method (OMP)  

2nd part refers to noise scale added to the imag. That is, constant in front of the random matrix:  
    -- *_noise_scale_*  
  
3rd part of the string says what are N1 and N2 respectively. These control the maximum order of  
shapelets included into the decomposition. For example:  
    -- *_20_20_* - this means N1 = 20 and N2 = 20  

4th part is about upper limit to the shapelet order that can be included:  
    -- *_n_max_*  
    
5th part is describing the coordinate system in which the decomposition was made. For example:  
    -- *_Polar_* - polar coordinate space for shapelets  
    -- *_XY_* - descartes coordinate space for shapelets  
    -- *_Elliptic_* - elipse coordinate space for shapelets (still not implemented)  

-------------

6th part:  
    For OMP:  
    -- *_number_of_shapelets_to_be_used_* - it describes the maximum number of shapelets used in OMP  
    For lasso:  
    -- *_alpha_* - coefficient in front of l_1 norm of the coefficients  
