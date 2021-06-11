from . Preconditioner import (Preconditioner,
    LeftPreconditioner, RightPreconditioner)
from . PreconditionerType import PreconditionerType

import scipy.sparse.linalg as spla
import scipy.sparse as sp
import numpy.linalg as npla
import numpy as np

class LeftILUT(PreconditionerType):

    def __init__(drop_tol=0.001, fill_factor=15):
        self.__init__()
        self.drop_tol = drop_tol
        self.fill_factor = fill_factor

    def form(self, A):
        return LeftILUTPreconditioner(A, drop_tol=self.drop_tol,
                                      fill_factor=self.fill_factor)


class RightILUT(PreconditionerType):

    def __init__(drop_tol=0.001, fill_factor=15):
        self.__init__()
        self.drop_tol = drop_tol
        self.fill_factor = fill_factor

    def form(self, A):
        return RightILUTPreconditioner(A, drop_tol=self.drop_tol,
                                      fill_factor=self.fill_factor)

class ILUTPreconditioner(Preconditioner):
    '''
    ILUT preconditioner implemented with SuperLU's incomplete factorization.
    '''
    def __init__(self, A, drop_tol=0.001, fill_factor=15):
        '''
        ILU Preconditioner constructor
        Constructor parameters:
        (*) A           -- The matrix to be approximately factored.
        (*) drop_tol    -- Tolerance for deciding whether to accept "fill"
                         entries in the approximate factorization. With a value
                         of 1, no fill is accepted. With a value of 0, all fill
                         is accepted. Default is 0.0001.
        (*) fill_factor -- Factor by which the number of nonzeros is allowed
                         to increase during the approximate factorization.
                         SuperLU's default is 10. With a low value of drop_tol
                         you may need to increase this.
        '''
        self._ILU = spla.spilu(A.tocsc(),
            drop_tol=drop_tol, fill_factor=fill_factor,
            diag_pivot_thresh=0.0)

    def ILU(self):
        return self._ILU


class LeftILUTPreconditioner(ILUTPreconditioner, LeftPreconditioner):
    '''ILU preconditioner applied from the left.'''
    def __init__(self, A, drop_tol=0.001, fill_factor=15):
        '''Constructor.'''
        super().__init__(A, drop_tol=drop_tol, fill_factor=fill_factor)
        super(ILUTPreconditioner, self).__init__()

    def applyLeft(self, vec):
        return self.ILU().solve(vec)


class RightILUTPreconditioner(ILUTPreconditioner, RightPreconditioner):
    '''ILU preconditioner applied from the right.'''
    def __init__(self, A, drop_tol=0.001, fill_factor=15):
        '''Constructor.'''
        super().__init__(A, drop_tol=drop_tol, fill_factor=fill_factor)
        super(ILUTPreconditioner, self).__init__()

    def applyRight(self, vec):
        return self.ILU().solve(vec)
