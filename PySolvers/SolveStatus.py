# ============================================================================
#
# Class SolveStatus collects information about the result of a linear solve.
#
# Katharine Long, Texas Tech University, 2021.
# ============================================================================

class SolveStatus:
    '''
    Encapsulation of the result of a linear or nonlinear solve attempt.
    '''
    def __init__(self, success, soln, resid, iters, msg=None):
        '''
        Constructor for SolverState object.
        '''
        self._success = success
        self._soln = soln
        self._resid = resid
        self._iters = iters
        self._msg = msg

    def success(self):
        '''Return a boolean indicating whether the solve succeeded.'''
        return self._success

    def soln(self):
        '''
        Return the solution. If the solve did not succeed, the value returned
        may be meaningless.
        '''
        return self._soln

    def resid(self):
        '''
        Return the residual. If the solve failed due to breakdown, the value
        returned will be meaningless.
        '''
        return self._resid

    def iters(self):
        '''
        Return the number of iterations used.
        '''
        return self._iters

    def msg(self):
        '''
        Return a description of the result.
        '''
        return self._msg

    def __str__(self):
        '''Return a string form of this status object.'''
        return 'SolverState(success={}, resid={}, iters={})'.format(
            self.success(), self.resid(), self.iters()
        )
