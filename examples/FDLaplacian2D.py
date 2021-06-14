import numpy as np
import numpy.linalg as la
import scipy.sparse as sp

def FDLaplacian2D(a, b, m):
    h = np.abs(b-a)/np.double(m+1)

    A = sp.dok_matrix((m*m,m*m))

    for ix in range(m):
        for iy in range(m):
            k = m*iy + ix
            A[k,k] = -4.0/h/h
            if iy > 0:
                A[k,k-m] = 1.0/h/h
            if iy < m-1:
                A[k,k+m] = 1.0/h/h
            if ix > 0:
                A[k,k-1] = 1.0/h/h
            if ix < m-1:
                A[k,k+1] = 1.0/h/h

    return A.tocsr()
