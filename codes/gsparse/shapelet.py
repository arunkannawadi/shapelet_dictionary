#import pdb; pdb.set_trace()
import math
import numpy
import numpy as np
from scipy.special import hermite as Hermite
from scipy.special import hermitenorm as HermiteNorm
from scipy.special import genlaguerre as GLaguerre
import gsparse

## For an excellent introduction to property decorators, see
## https://www.programiz.com/python-programming/property

## I use property and setter extensively, so that one doesn't
## change the type of the attributes during the progress.
## Setter ensures type-checking everytime, rather than during initialisation only

class Shapelet(object):
    ## Inherit from object so that we can have read-only attirbutes
    def __init__(self,*args,**kwargs):
        ## Parse the args
        if len(args)>2:
            raise TypeError("Too many unnamed arguments to the Shapelet constructor")
        elif len(args)==2:
            if isinstance(args[0],int) and isinstance(args[1],int):
                order = (args[0],args[1])
            else:
                raise TypeError("Shapelet must be constructed with two integers")
        elif len(args)==1:
            if hasattr(args,'__iter__'):
                if len(args[0])==2:
                    if isinstance(args[0][0],int) and isinstance(args[0][1],int):
                        order = args
                    else:
                        raise TypeError("Shapelet must be constructed with a\
                                         tuple with two integers")
                else:
                    raise TypeError("Shapelet must be constructed with a\
                                     tuple of length 2")
            else:
                raise TypeError("Shapelet must be constructed with a list or\
                                 a tuple")
        elif len(args)==0:
            raise TypeError("The order of the shapelet must be provided.")

        self._order = order

        # Parse the kwargs
        default_cs = 'polar'
        if not 'coordinate' in kwargs:
            self._cs = default_cs
        else:
            cs = kwargs['coordinate']
            if cs is None:
                self._cs = default_cs
            else:
                self._cs = cs

            if not (cs=='cartesian' or cs=='polar'):
                import warnings
                warnings.warn('coordinate not recognised. Setting %s' % default_cs)
                self._cs = default_cs
            else:
                self._cs = cs

        if not 'shape' in kwargs:
            shape = gsparse.Shape()
        else:
            shape = kwargs['shape']
        self.shape = shape

        if not 'center' in kwargs:
            center = np.array([0.,0.])
            self.center = center
        else:
            center = kwargs['center']
            if not len(center)==2:
                raise TypeError("'center' has to be a tuple of length 2")
            elif not (isinstance(center[0],float) and isinstance(center[1],float)):
                raise TypeError("'center' has to be a tuple of two floats")
            else:
                self.center = center

        self._prefactor = 1. ## determines the normalisation

    def __repr__(self):
        return "gsparse.Shapelet(cs=%s,order=(%d,%d),center=(%.2f,%.2f),shape=%s)"\
                % (self.cs,self.order[0],self.order[1],self.center[0],self.center[1],self.shape)

    def __idiv__(self,f):
        self._prefactor /= f
        return self

    def __div__(self,f):
        new = self.copy()
        new /= f
        return new

    def copy(self):
        """ Make a deep copy of the instance.
        """
        import copy
        return copy.deepcopy(self)

    @property
    def cs(self):
        """ Get Coordinate system (cs) of the shapelets.

        cs is a read-only attribute. It is set during the instantiation of an object.
        Possible values are 'cartesian' and 'polar' [default]
        """
        #print "Getting cs"
        return self._cs
    #@cs.setter
    #def cs(self,coord_sys):
    #    print "Setting cs"
    #    if coord_sys=='cartesian' or coord_sys=='polar':
    #        self._cs = coord_sys
    #    else:
    #        import warnings
    #        warnings.warn("Invalid value for cs. Setting 'cartesian' by default")
    #        self._cs = 'cartesian'
    ## Coordinate system (cs) is a read-only attribute, and set during initialisation
    ## No setter function defined

    @property
    def shape(self):
        """ Corresponding gsparse.Shape instance.
        """
        #print "Getting shape"
        return self._shape
    @shape.setter
    def shape(self,shape):
        #print "Setting shape"
        if not isinstance(shape,gsparse.Shape):
            import warnings
            warnings.warn("'shape' argument to Shapelet must a\
                        Shape instance. Leaving it unchanged.")
        else:
            self._shape = shape

    @property
    def order(self):
        return self._order

    @property
    def center(self):
        return self._center
    @center.setter
    def center(self,center):
        _valid_center(center,'center')
        self._center = center

    def stretch(self,*args,**kwargs):
        """ Isotropically stretch the shapelet(s).

            @param strech_factor    The factor by which the size is scaled.
                                    [default: 1.0]
        """
        self.shape.stretch(*args,**kwargs)

    def rotate(self,*args,**kwargs):
        """ Rotate the shapelet(s).

            @param theta    The angle (in radians, CCW) by which the shapelet(s) must be rotated.
                            [default: 0.0]
        """
        self.shape.rotate(*args,**kwargs)

    def shear(self,*args,**kwargs):
        """ Shear the shapelet(s).

            @param q    The axis-ratio which the base Gaussian must have.
                        [default: 1.0]
        """
        self.shape.shear(*args,**kwargs)

    def translate(self,offset):
        _valid_center(offset,'offset')
        self.center += offset

    def draw(self,X,Y):
        if not X.shape==Y.shape:
            raise ValueError("The dimensions of 'X' and 'Y' must match.")
        x, y = X.flatten(), Y.flatten()
        ## Apply the center
        x -= self.center[0]
        y -= self.center[1]

        XY = np.vstack((x,y))
        UV = (self.shape.matrix)*XY
        U,V = UV

        if self.cs=='cartesian':
            m,n = self.order
            He_m, He_n = HermiteNorm(m), HermiteNorm(n)
            ## To have easy element-wise multiplication, we deal with arrays
            z = He_m(U.A)*He_n(V.A)*np.exp(-(U.A**2+V.A**2)/2)
        else:
            p,q = self.order
            N,M = p+q, abs(p-q)

            r2 = (U.A**2+V.A**2)
            r = np.sqrt(r2)
            phi = np.zeros_like(r)
            for k in xrange(X.size):
                #print "Uk,Vk", U.A[0][k], V.A[0][k]
                phi[0][k] = math.atan2(V.A[0][k],U.A[0][k])
            ## phi is in radians

            Lq = GLaguerre(n=q,alpha=M)
            if p>=q:
                trigfactor = np.cos(M*phi)
            else:
                trigfactor = np.sin(M*phi)

            z = self._prefactor*trigfactor*Lq(r2)*np.exp(-r2/2.)*(r**M)

        Z = np.reshape(z,X.shape)
        return Z


class CartesianShapelet(Shapelet):
    def __init__(self,m,n,**kwargs):
        kwargs['coordinate'] = 'cartesian'
        args = (m,n)
        Shapelet.__init__(self,*args,**kwargs)

    def __repr__(self):
        return "gsparse.CartesianShapelet(m=%d,n=%d,shape=%s)" % (self.order[0],self.order[1],self.shape)

    def _draw(self,X,Y):
        if not X.shape==Y.shape:
            raise ValueError("The dimensions of 'X' and 'Y' must match.")
        x, y = X.flatten(), Y.flatten()
        ## Apply the center
        x -= self.center[0]
        y -= self.center[1]

        XY = np.vstack((x,y))
        UV = (self.shape.matrix)*XY
        U,V = UV

        m,n = self.order
        He_m, He_n = HermiteNorm(m), HermiteNorm(n)
        ## To have easy element-wise multiplication, we deal with arrays
        z = He_m(U.A)*He_n(V.A)*np.exp(-(U.A**2+V.A**2)/2)
        Z = np.reshape(z,X.shape)
        return Z


class PolarShapelet(Shapelet):
    def __init__(self,p,q,**kwargs):
        kwargs['coordinate'] = 'polar'
        args = (p,q)
        Shapelet.__init__(self,*args,**kwargs)

    def __repr__(self):
        return "gsparse.PolarShapelet(p=%d,q=%d,shape=%s)" % (self.order[0],self.order[1],self.shape)

    def _draw(self,X,Y):
        if not X.shape==Y.shape:
            raise ValueError("The dimensions of 'X' and 'Y' must match.")
        x, y = X.flatten(), Y.flatten()
        ## Apply the center
        x -= self.center[0]
        y -= self.center[1]

        XY = np.vstack((x,y))
        UV = (self.shape.matrix)*XY
        U,V = UV

        p,q = self.order
        N,M = p+q, abs(p-q)

        r2 = (U.A**2+V.A**2)
        r = np.sqrt(r2)
        phi = np.zeros_like(r)
        for k in xrange(X.size):
            #print "Uk,Vk", U.A[0][k], V.A[0][k]
            phi[0][k] = math.atan2(V.A[0][k],U.A[0][k])
        ## phi is in radians

        Lq = GLaguerre(n=q,alpha=M)
        if p>=q:
            trigfactor = np.cos(M*phi)
        else:
            trigfactor = np.sin(M*phi)

        prefactor = 1. ## to be filled in
        z = prefactor*trigfactor*Lq(r2)*np.exp(-r2/2.)*(r**M)
        Z = np.reshape(z,X.shape)
        return Z

## Convenience functions
def _valid_center(center,var_name):
    if not isinstance(center,numpy.ndarray):
        raise TypeError("%s must be an array" % var_name)
    if not center.shape==(2,):
        raise TypeError("%s must be an array of length 2" % var_name)
    if not center.dtype in ['int','int64','float','float64']:
        raise TypeError("%s must be an array of float type" % var_name)
    return True
