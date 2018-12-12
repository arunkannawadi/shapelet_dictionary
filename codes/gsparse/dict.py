import numpy
import numpy as np
import gsparse

class Dictionary(gsparse.Shapelet,object):
    def __init__(self,max_order,**kwargs):
        self.shape = gsparse.Shape(**kwargs)
        if 'center' in kwargs.keys() and kwargs['center'] is not None:
            self.center = kwargs['center']
        else:
            self.center = np.array([0.,0.])
        # The above 'declares' the variable, after which Shapelet routines take over
        self.max_order = max_order
        self._shapelets = [ ]
        self.populate()

        ## The below will be filled after drawing
        self.matrix = None
        self._X = None
        self._Y = None

    def __repr__(self):
        return "gsparse.Dictionary(shape=%s,max_order=%d,center=(%.2f,%.2f))"\
                % (self.shape,self.max_order, self.center[0],self.center[1])

    def __idiv__(self,f):
        for sl in self.shapelets:
            sl /= f
        ## Access the matrix without redrawing
        if self.matrix is not None:
            self.matrix /= f
    ## __div__ is inherited from shapelet
    ## __copy__ is inherited from shapelet

    #def copy(self):
    #    D = Dictionary(self.max_order,center=self.center)
    #    D.shape = self.shape
    #    D._shapelets = self.shapelets # use private variable, since this shouldn't be done otherwise
    #    if self.getX() is not None and self.getY() is not None:
    #        D.draw(self.getX(),self.getY()) # to fill X,Y and image matrix
    #    return D

    @property
    def shape(self):
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
    def shapelets(self):
        return self._shapelets
    @property
    def cs(self):
        return self.shapelets[0].cs

    @property
    def max_order(self):
        return self._max_order
    @max_order.setter
    def max_order(self,max_order):
        if max_order<0:
            raise ValueError("'max_order' has to be non-negative")
        if not isinstance(max_order,int):
            print type(max_order)
            raise TypeError("'max_order' has to be an integer")
        self._max_order = max_order

    def getX(self):
        return self._X
    def getY(self):
        return self._Y
    def getMatrix(self,redraw=True):
        if redraw is True:
            if self.getX() is None or self.getY() is None:
                raise AttributeError("Input grid is not yet supplied.")
            else:
                self.redraw()
        return self.matrix

    def include(self,order):
        shapelet_orders = [shapelet.order for shapelet in self.shapelets]
        if not (order[0],order[1]) in shapelet_orders:
            shapelet = gsparse.Shapelet(*order,shape=self.shape)
            self._shapelets.append(shapelet)

    def exclude(self,order):
        shapelet_orders = [shapelet.order for shapelet in self.shapelets]
        try:
            idx = shapelet_orders.index((order[0],order[1]))
            self._shapelets.pop(idx)
        except ValueError:
            pass ## silently do nothing

    def populate(self):
        N = self.max_order
        for m in xrange(0,N+1):
            for n in xrange(0,1+N-m):
                self.include((m,n))

    def draw(self,X,Y):
        M = np.empty((X.size,len(self.shapelets)))
        for idx,shapelet in enumerate(self.shapelets):
            Z = shapelet.draw(X,Y)
            M[:,idx] = Z.flatten()

        self.matrix = M
        self._X = X
        self._Y = Y

    def redraw(self):
        if self.getX() is not None and self.getY() is not None:
            self.draw(self.getX(),self.getY())

    def normalise(self,p=2):
        M = self.getMatrix()
        Mp = np.abs(M)**p ## element-wise raised to power
        pnorm = (Mp.sum(axis=0))**(1./p)
        self.matrix = M/pnorm
        ## Reflext the change in the shapelets too, so that redrawing doesn't destroy normalization
        for i,sl in enumerate(self.shapelets):
            sl /= pnorm[i]

    def getCoherence(self):
        M = self.getMatrix()
        ## Do not call normalise method
        norm = np.sqrt( (np.abs(M)**2).sum(axis=0) )
        Mnorm = M/norm
        G = np.dot(Mnorm.T,Mnorm)
        return np.abs((G-np.identity(G.shape[0]))).max()

    def visualise(self,filename=None,grid=True,cmap=None,interpolation='nearest'):
        import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        if cmap is None:
            cmap = cm.bwr

        fig, axes = plt.subplots(self.max_order+1,self.max_order+1)
        fig.set_size_inches((self.max_order+1)*9,(self.max_order+1)*9)

        M = self.getMatrix()
        orders = [sl.order for sl in self.shapelets]
        for i in xrange(len(orders)):
            m,n = orders[i]
            print 'Working on', m,n
            ax = axes[m,n]

            z = np.reshape(M[:,i],self.getX().shape)
            zmin, zmax = z.min(), z.max()
            vmax = max(abs(zmax),abs(zmin))
            vmin = -vmax
            _im = ax.imshow(z,cmap=cmap,interpolation=interpolation,vmin=vmin,vmax=vmax)
            _cbar = plt.colorbar(_im,ax=ax, fraction=0.046, pad=0.04)
            #https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
            ax.set_title(str((m,n)))
            ax.grid(grid)

        #fig.suptitle('offset=(%.1f,%.1f),q=%.2f,theta=%.4f,stretch=%.2f' % (offset[0],offset[1],q,theta,stretch_factor) )
        #dirname = '/Users/arunkannawadi/Dropbox/Shapelets/code/scripts/output/'
        #filename = 'visualise_%s.pdf' % cs
        if filename is not None:
            fig.savefig(filename)
        plt.show()

class CompoundDictionary(Dictionary,object):
    def __init__(self,*args,**kwargs):
        self._dicts = [ ]
        self._shapelets = [ ]
        ## A list of all the shapes
        self._shapes = [ ]
        self.max_orders = [ ]
        self.coord = [ ]
        self._orders = []

        ## A unique list of all the shapes available
        self.shapes = list(set(self._shapes))

        ## The following take legit values after evaluate is called
        self._X = None
        self._Y = None
        self.matrix = None

    def __repr__(self):
        return "gsparse.CompoundDictionary(%s)" % self.dicts

    @property
    def dicts(self):
        return self._dicts

    @property
    def shapelets(self):
        S = [ ]
        for di in self.dicts:
            S += di.shapelets
        return S

    @property
    def cs(self):
        return set([sl.cs for sl in self.shapelets])

    @property
    def shapes(self):
        return [di.shape for di in self.dicts]
    @shapes.setter
    def shapes(self,new_shapes):
        self._shapes += new_shapes
    #draw routine is inherited
    #redraw routine is inherited

    def rotate(self,theta,dict_id=None):
        if dict_id is None: # rotate all
            for idx in xrange(len(self.dicts)):
                self.dicts[idx].rotate(theta)
        elif isinstance(dict_id,int):
            if dict_id<0:
                raise ValueError("The dictionary id has to be a non-negative integer")
            self.dicts[dict_id].rotate(theta)

    def stretch(self,stretch_factor,dict_id=None):
        if dict_id is None: # rotate all
            for idx in xrange(len(self.dicts)):
                self.dicts[idx].stretch(stretch_factor)
        elif isinstance(dict_id,int):
            if dict_id<0:
                raise ValueError("The dictionary id has to be a non-negative integer")
            self.dicts[dict_id].stretch(stretch_factor)

    def shear(self,q,dict_id=None):
        if dict_id is None: # rotate all
            for idx in xrange(len(self.dicts)):
                self.dicts[idx].shear(q)
        elif isinstance(dict_id,int):
            if dict_id<0:
                raise ValueError("The dictionary id has to be a non-negative integer")
            self.dicts[dict_id].shear(q)

    def translate(self,offset,dict_id=None):
        if dict_id is None: # rotate all
            for idx in xrange(len(self.dicts)):
                self.dicts[idx].translate(offset)
        elif isinstance(dict_id,int):
            if dict_id<0:
                raise ValueError("The dictionary id has to be a non-negative integer")
            self.dicts[dict_id].translate(offset)

    def include(self,dictionary):
        if not dictionary in self.dicts:
            self.dicts.append(dictionary)

    def visualise(self):
        print "Visualise individual dictionaries independenly."
