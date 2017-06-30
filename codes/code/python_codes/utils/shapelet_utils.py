import numpy as np

def get_max_order(basis, n_max):
    """
    Calculating max order of shapelets included in order to get
    a symmetric triangle
    
    @param basis Basis of the decomp.
    @param n_max Chosen max number by user

    ---------------
    For example

    n_max = 5

    XY
    return 1 -----> Combinations of shapelets available (0,0), (0,1), (1,0)
    // If 2 is included the number of combinations would exceed n_max and 
    // triangle won't be symmetric
    return 1 -----> Combinations of shapelets (0,0), (-1,1), (1,1)
    // Same as above
    """

    #xy_max_order = lambda n: int(np.floor(np.sqrt(n) - 1))
    #polar_max_order = lambda n: int(np.floor( (-3 + np.sqrt(9 + 8*(n-1)))/2.))
    
    ## Get the symmetric triangle for XY and same for Polar
    ## It's the same because for XY I look at the m+n, and for Polar just n
    max_order = lambda n: int(np.floor( (-3 + np.sqrt(9 + 8*(n-1)))/2.))

    return max_order(n_max)

def sum_max_order(basis, n):
    """
    Given the maximum order of shapelet included
    calculate how many combinations of shapelets there are
    
    -------------
    For example:
    n = 1
    
    XY basis / Elliptical basis
    return 3 ---> Combinations (0,0), (0,1), (1,0) 
    // Because in the decomp I hace conditon n+m<= 1, in that way i get symmetric triangle
    Polar
    return 3 ---> Combinations (0,0), (-1,1), (1,1) 
    // If 2 would be included then the order of shapelet would exceed 1 
    """
    return n*(n+1)/2 + n + 1
