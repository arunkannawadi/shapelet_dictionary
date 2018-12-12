import numpy
## Following are different ways to solve Ax = b, approximately.

def solve(A, b, solver=None, **kwargs):
    """
    Solve the equation Ax = b. If an exact solution is not possible, find an
    approximate solution using any of the available methods (see the list of
    available methods under the 'solver' parameter)

    @param A         The coefficient matrix A
    @param b         The RHS of the linear equation
    @param solver    The solver to use. This has to be one of
                     'exact','lstsq','omp','lasso'
    """

    if isinstance(A, numpy.ndarray):
        A = numpy.matrix(A)

    if solver=='exact':
        if  A.shape[0]==A.shape[1] and np.linalg.matrix_rank(A)==A.shape[0]:
            x = _solver_exact(A,b)
        else:
            raise ValueError("'A' is not a full-rank square matrix. I cannot find an exact solution")
    elif solver=='lstsq':
        x = _solver_lstsq(A,b)
    elif solver=='omp':
        K = kwargs['K']
        x = _solver_OMP(A,b,K)
    elif solver=='lasso':
        alpha = kwargs['alpha']
        x = _solver_Lasso(A,b,alpha)

    ## Calculate the residual
    print A.shape, A.transpose().shape, x.shape, b.shape
    r = b - x*A.transpose()

    return x,r

def _solver_exact(A, b):
    x = numpy.linalg.inv(A)*b
    return x

def _solver_lstsq(A, b):

    """ Standard least-squares algorithm, with no free parameters.
    """

    soln = numpy.linalg.lstsq(A,b)
    x = soln[0]

    return x

def _solver_OMP(A, b, K):
    """
    Find a K-sparse solution to Ax = b.

    @param K    Sparsity of the solution.
    """

    from sklearn.linear_model import OrthogonalMatchingPursuit as OMP
    omp = OMP(n_nonzero_coefs=K)
    omp.fit(A,b)
    x = omp.coef_

    return x

def _solver_Lasso(A,b,alpha):
    """
    Lasso, L1-solver
    """
    from sklearn import linear_model
    model = linear_model.Lasso(alpha=alpha,max_iter=10000,fit_intercept=False)
    lasso = model.fit(A,b)
    x = lasso.coef_

    return x
