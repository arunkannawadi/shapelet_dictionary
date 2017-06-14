import numpy.linalg as linalg
import numpy as np

k = 2.52
n = 1.24

x = 10*np.random.rand(20)
y = k*x+ n

sigma_x = 10*np.random.rand(20)
sigma_y = 2*np.random.rand(20)

x_data = np.zeros_like(x)
y_data = np.zeros_like(y)
A = np.zeros((20,2))
z = 1.

for i in xrange(len(x)):
    x_data[i] = x[i] + z*sigma_x[i]
    y_data[i] = y[i] + z*sigma_y[i]
    z *= -1.

A[:,0] = x_data
A[:,1] = np.ones(len(x_data))

U, s, VT = linalg.svd(A, full_matrices = True, compute_uv = True)

V = VT.transpose()
S_dual = np.eye(20,2)*(1./s)

coeffs = np.dot(V, np.dot(S_dual.transpose(),np.dot(U.transpose(), y_data)))
print(coeffs)

coeffs_lstsq = linalg.lstsq(A,y_data)[0]
print(coeffs_lstsq)
