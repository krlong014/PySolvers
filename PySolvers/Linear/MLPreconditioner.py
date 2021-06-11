from . Preconditioner import Preconditioner
from . SmoothedAggregation import SmoothedAggregationMLSequence
from . VCycleManager import VCycleManager
from . ClassicSmoothers import JacobiSmoother, GaussSeidelSmoother

import scipy.sparse.linalg as spla
import scipy.sparse as sp
import numpy.linalg as npla
import numpy as np

class AMGPreconditioner(Preconditioner):
    '''
    Smoothed aggregation based multilevel preconditioner
    '''
    def __init__(self, A, right=True, numLevels=2, nuPre=2, nuPost=2,
                smoother=GaussSeidelSmoother):
        '''
        AMG preconditioner constructor
        Constructor parameters:
        (*) A           -- The matrix to be approximately factored.
        '''
        self._right = right
        seq = SmoothedAggregationMLSequence(A, numLevels=numLevels)
        self._cycleMgr=VCycleManager(seq, nuPre=nuPre, nuPost=nuPost, smoother)


    def applyLeft(self, vec):
        '''Apply from the left.'''
        if self._right:
            return vec
        else:
            return self._apply(vec)


    def applyRight(self, vec):
        '''Apply from the right.'''
        if not self._right:
            return vec
        else:
            return self._apply(vec)


    def _apply(self, vec):
        '''Apply the multilevel operator to a vector.'''
        x = np.copy(vec)
        return self._cycleMgr.runCycle(vec, x))
