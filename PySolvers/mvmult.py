import scipy.sparse as sp
import numpy as np

'''
Unified mvmult user interface for both scipy.sparse and numpy matrices.
In scipy.sparse, mvmult is done using the overloaded * operator, e.g., A*x.
In numpy, mvmult is done using the dot() function, e.g., dot(A,x).
This function chooses which to use based on whether or not A is stored as
a sparse matrix.
'''

def mvmult(A, x):
    '''Multiply a matrix A times a vector x.'''

    if sp.issparse(A):
        return A*x
    else:
        return np.dot(A,x)
