import numpy
import gsparse
from solver_routines import solve as gensolve

class Signal(object):
    def __init__(self,image,D,solver=None,**kwargs):
        if isinstance(D,gsparse.Dictionary):
            self.D = gsparse.CompoundDictionary()
            self.D.include(D.copy())
            self.D.draw(D.getX(),D.getY())
        elif isinstance(D,gsparse.CompoundDictionary):
            self.D = D.copy()
        else:
            raise TypeError("'D' must be an instance of gsparse.CompoundDictionary")

        if not image.size==self.D.getMatrix(redraw=False).shape[0]:
            raise ValueError("The input image is %d dimensional whereas D is %d \
                                dimensional" % (image.size,D.getMatrix(redraw=False).shape[0]))
        self.image = image
        if solver is not None:
            self.solve(solver,**kwargs)

    def solve(self,solver,**kwargs):
        self.solver = solver
        self.solver_params = kwargs
        self.code, r = gensolve(self.D.getMatrix(redraw=False),self.image.flatten(),\
                                    solver=solver,**kwargs)
        self.residual = numpy.reshape(r,self.image.shape)
        self.signal = self.image - self.residual

    def count_nonzero(self):
        """ Count the number of non-zero coefficients in the representation.
        """
        return numpy.linalg.norm(self.code,ord=0)

    def compress(self,*args,**kwargs):
        """ Get rid of small coefficients, as specified by the user.
        """
        flag = (numpy.abs(self.coefficients)>min_coeff)
        self.coefficients = np.array(self.coefficients)[flag].tolist()
        ## Yet to truncate the Dictionary

    def truncate(self):
        """ Shed the zero coefficients and the corresponding atoms.
        """
        self.compress(min_coeff=0)

    def draw(self):
        self.truncate()
        D = self.dictionary.draw()
        y = D*self.coefficients
        return y
