from . Preconditioner import RightPreconditioner

import scipy.sparse.linalg as spla
import scipy.sparse as sp
import numpy as np

class ICRightPreconditioner(RightPreconditioner):
    '''Incomplete Cholesky preconditioner, to be applied from the right.'''

    def __init__(self, A, drop_tol=0.001, fill_factor=15):
        '''Constructor.'''
        super().__init__()

        # Construct the ILU approximate factorization using SuperLU accessed
        # through the scipy.sparse.linalg interface. Super LU needs
        # to use CSC format, so do the conversion here.
        # To adapt SuperLU to incomplete Cholesky, we need to disable
        # column permutation and row pivoting. We then scale U by sqrt(D^-1)
        # get the Cholesky factor L^T.

        ILU = spla.spilu(A.tocsc(),
            drop_tol=drop_tol, fill_factor=fill_factor,
            diag_pivot_thresh=0.0, options={'ColPerm':'NATURAL'})
        n = A.shape[0]
        diagScale = np.reciprocal(np.sqrt(ILU.U.diagonal()))
        DInv = sp.dia_matrix((diagScale,[0]), shape=(n,n))

        self._Lt = DInv * ILU.U
        del ILU # delete this now to save memory
        self._L = self._Lt.transpose()
        self._Lt = self._Lt.tocsr()
        self._L = self._L.tocsr()

    def applyRight(self, vec):
        '''Apply the preconditioner from the right.'''
        # Apply L^{-1}
        u = spla.spsolve_triangular(self._L, vec, lower=True)
        # Apply L^{-T}
        return spla.spsolve_triangular(self._Lt, u, lower=False)
