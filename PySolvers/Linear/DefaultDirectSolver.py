# ============================================================================
#
# Default direct solver objects:
#
# Class DefaultDirect is a factory class for user-level specification that
# a direct solver is to be used.
#
# Class DefaultDirectSolver provides a solve() function that calls either
# numpy's default direct solver or scipy's default sparse direct solver.
#
# Katharine Long, Texas Tech University, 2021.
# ============================================================================

import numpy as np
import scipy.sparse as sp
import numpy.linalg as npla
import scipy.sparse.linalg as spla

from . LinearSolver import LinearSolver, LinearSolverType
from .. SolveStatus import SolveStatus


class DefaultDirect(LinearSolverType):
    '''
    Class DefaultDirect is a factory class for user-level specification that
    a direct solver is to be used.
    '''
    def __init__(self, name='Default direct'):
        '''Constructor.'''
        super().__init__(name=name)


    def makeSolver(self, name=None):
        '''Creates a default direct solver object.'''
        useName = name
        if useName==None:
            useName = self.name()
        return DefaultDirectSolver(name=useName)

class DefaultDirectSolver(LinearSolver):
    '''
    Class DefaultDirectSolver provides a solve() function that calls either
    numpy's default direct solver or scipy's default sparse direct solver.
    '''
    def __init__(self, name='Default direct'):
        '''Constructor.'''
        super().__init__(name=name)

    def solve(self, A, b):
        '''Solve A*x=b for x. Returns a SolveStatus object.'''

        # Get size of matrix
        n,nc = A.shape
        # Make sure matrix is square
        assert(n==nc)
        # Make sure A and b are compatible
        assert(n==len(b))

        # Use dense or sparse direct solver as determined by type
        # of matrix.
        try:
            if sp.isspmatrix(A):
                soln = spla.spsolve(A, b)
            elif isinstance(A, np.ndarray):
                soln = npla.solve(A, b)
            else:
                return SolveStatus(False, None, None, None,
                                   'Input to solver [%s] not numpy or scipy' %
                                   self.name())
            return SolveStatus(True, soln, None, None,
                               '%s solve succeeded' % self.name())
        except Exception as ex:
            return SolveStatus(False, None, None, None,
                               '{} solve failed: {}'.format(self.name(), ex))
