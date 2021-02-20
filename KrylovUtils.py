# Some helpful functions for Krylov methods
# Katharine Long, Texas Tech.
# This code is in the public domain.

import time
import numpy as np
import scipy.sparse as sp


# A simple timer class
class MyTimer:
    def __init__(self, name):
        self.name = name
        self.start = time.time()

    def stop(self):
        self.stop = time.time()

    def walltime(self):
        return self.stop - self.start

# Unified mvmult user interface for both scipy.sparse and numpy matrices.
# In scipy.sparse, mvmult is done using the overloaded * operator, e.g., A*x.
# In numpy, mvmult is done using the dot() function, e.g., dot(A,x).
# This function chooses which to use based on whether A is stored as
# a sparse matrix.
def mvmult(A, x):
    if sp.issparse(A):
        return A*x
    else:
        return np.dot(A,x)

# Check whether a matrix is square. Raise an exception if not.
def checkSquare(A, name):
    n,nc = A.shape
    if n!=nc:
        raise RuntimeError('%s detected non-square matrix; size is %d by %d'
            % (name, n, nc))
