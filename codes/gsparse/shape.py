import numpy
import numpy as np
import gsparse

class Shape(object):
    ## A fancy way of treating 2x2 matrices, with convenient routines
    """@file shape.py

    A Class for describing 2x2 matrices, with convenient routines

    The Shape class
    """
    def __init__(self,q=1.,theta=0.,stretch_factor=1.,**kwargs):
        if 'matrix' in kwargs.keys():
            self.matrix = kwargs['matrix']
        else:
            self.matrix = numpy.matrix(np.eye(2)) # 2x2 Identity
            self.transformation(q,theta,stretch_factor)

    def __repr__(self):
        #return "gsparse.Shape(%s)" % self.matrix
        return "gsparse.Shape([[%f,%f],[%f,%f]])"\
             % (self.matrix[0,0],self.matrix[0,1],self.matrix[1,0],self.matrix[1,1])

    def copy(self):
        import copy
        return copy.deepcopy(self)

    @property
    def matrix(self):
        return self._matrix
    @matrix.setter
    def matrix(self,M):
        if isinstance(M, gsparse.Shape):
            M = M.matrix
        elif isinstance(M, numpy.ndarray):
            M = numpy.matrix(M)
        elif not isinstance(M, numpy.matrix):
            raise TypeError("'matrix' must be a numpy.matrix instance")

        if not M.shape==(2,2):
            raise ValueError("'matrix' must be a 2x2 numpy.matrix instance")

        self._matrix = M

    def transform(self,T):
        """
        Apply a generic linear transformation T to the shape, so that
        Shape S --> T S. The general form of T = (uRS)^(-1) = invS invR / u.

        Commonly used transformations such as stretching, rotation, shear
        can be applied through the corresponding routines: 'stretch', 'rotate', 'shear'.
        """
        if not isinstance(T,numpy.matrixlib.defmatrix.matrix):
            T = numpy.matrix(T)

        if not T.shape==(2,2):
            raise ValueError("The transformation 'T' must be a 2x2 numpy.matrix instance")

        #self._matrix = numpy.linalg.inv(T)*self._matrix
        self.matrix = T*self.matrix
        #self._matrix = T*self._matrix

    def stretch(self,stretch_factor):
        """
        Isotropically stretch the shape.
        """
        if stretch_factor==0:
            raise ValueError("The 'stretch_factor' cannot be zero.")
        elif stretch_factor<0:
            import warnings
            warnings.warn("The 'stretch_factor' is negative! Proceeding anyway ...")
        T = (1./stretch_factor)*np.matrix(np.eye(2))
        self.transform(T)
        #self._matrix *= stretch_factor

    def rotate(self,theta):
        """
        Rotate the shape by `theta' radians.
        """
        c = numpy.cos(theta)
        s = numpy.sin(theta)
        invR = np.matrix([[c, -s],[s,c]]) # Rotation matrix inverse
        M = self.matrix # existing shape matrix
        T = M*invR*numpy.linalg.inv(M) # new shape matrix
        self.transform(T)

    def shear(self,q):
        if not q>0:
            raise ValueError("The 'q' argument to 'shear' must be positive")

        #M = self.matrix # existing shape matrix
        invS = np.matrix([[np.sqrt(q),0.],[0.,1./np.sqrt(q)]]) ## inverse shear
        #T = M*invS*numpy.linalg.inv(M) # new shape matrix
        self.transform(invS)

    def transformation(self,q=1.,theta=0.,stretch_factor=1.):
        self.rotate(theta)
        self.shear(q)
        self.stretch(stretch_factor)
