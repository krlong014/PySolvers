# ============================================================================
#
# Class CommonSolverArgs collects typical control parameters for linear
# and nonlinear solvers. The user will normally set solver parameters
# through this object.
#
# Class IterativeSolver is a base class for linear solvers that handles
# common maintenance jobs.
#
# Katharine Long, Texas Tech University, 2020-2021.
# ============================================================================


from abc import ABC, abstractmethod
import numpy as np
import scipy.sparse as sp
import numpy.linalg as npla
from PyTab import Tab
from . SolveStatus import SolveStatus
from . NamedObject import NamedObject

# -----------------------------------------------------------------------------
# CommonSolverArgs

class CommonSolverArgs:
    '''
    Class CommonSolverArgs ollects typical control parameters for iterative
    linear or nonlinear solvers. The user will normally set solver parameters
    through this object.

    * Attributes:
        * maxiter -- maximum number of iterations allowed before stopping.
        * failOnMaxiter -- whether reaching maxiter is considered a failure.
        This is normally true.
        * tau -- relative residual tolerance.
        * norm -- the norm that will be used in checking for convergence
        * showIters -- whether to print out iteration status
        * interval -- number of iterations between output of status
        * showFinal -- whether to print convergence or error information upon
        termination
    '''
    def __init__(self,
                maxiter=100,
                failOnMaxiter=True,
                tau=1.0e-8,
                norm=npla.norm,
                showIters=True,
                showFinal=True,
                interval=1):
        '''Constructor'''
        self.maxiter = maxiter
        self.failOnMaxiter = failOnMaxiter
        self.tau = tau
        self.norm = norm
        self.showIters = showIters
        self.showFinal = showFinal
        self.interval = interval


# IterativeSolver

class IterativeSolver(NamedObject):
    '''
    Class IterativeSolver is a base class for linear and nonlinear solvers
    # that handles common parameters and common maintenance jobs.
    '''
    def __init__(self, control, name=''):
        super().__init__(name)
        self._control = control

    def maxiter(self):
        '''Return the maximum number of iterations allowed.'''
        return self._control.maxiter

    def failOnMaxiter(self):
        '''Indicate whether reaching maxiter is to be considered a failure.'''
        return self._control.failOnMaxiter

    def tau(self):
        '''Return the relative residual tolerance.'''
        return self._control.tau

    def setTolerance(self, tau):
        self._control.tau = tau

    def norm(self, x):
        '''Evaluate the norm of a vector x using the solver's specified norm.'''
        return self._control.norm(x)

    def reportIter(self, iter, normR, normR0):
        '''
        If control.showIters==True, print information about the current iteration
        at intervals specified by the control.interval parameter. Otherwise,
        do nothing.
        '''
        tab = Tab()
        if self._control.showIters and (iter % self._control.interval)==0:
            print('%s%s iter=%7d ||r||=%12.5g ||r||/r0=%12.5g' %
                  (tab, self.name(), iter, normR, normR/normR0))

    def handleConvergence(self, iter, x, normR, normB):
        '''
        Deal with detection of convergence (or at least successful stopping).
        Returns a successful SolveStatus.
        '''
        self.reportSuccess(iter+1, normR, normB)
        return SolveStatus(success=True, iters=iter+1, soln=x, resid=normR)

    def handleBreakdown(self, iter, msg):
        '''
        Deal with breakdown.
        '''
        self.reportBreakdown(msg=msg)
        return SolveStatus(success=False, iters=iter,
                           soln=None, resid=None, msg=msg)

    def handleMaxiter(self, iter, x, normR, normB):
        '''
        Deal with reaching maxiter. Returns a SolveStatus indicating either
        failure (most common) or success (if failOnMaxiter==False).
        '''
        # If we're here, maxiter has been reached. This is normally a failure.
        if self.failOnMaxiter():
            self.reportFailure(iter, normR, normB)
            return SolveStatus(success=False, iters=iter, soln=x, resid=normR,
                            msg='failure to converge')
        else: # Termination at maxiter is OK, as in use as preconditioner
            self.reportSuccess(iter+1, normR, normB)
            return SolveStatus(success=True, iters=iter, soln=x, resid=normR)

    def reportSuccess(self, iter, normR, normB):
        '''If control.showFinal==True, print a message upon convergence.'''
        tab=Tab()
        if self._control.showFinal:
            normRel = normR
            if normR != 0:
                normRel = normR/normB
            print('%s%s solve succeeded: iters=%7d, ||r||/r0=%12.5g' %
                (tab, self.name(), iter, normRel))

    def reportBreakdown(self, msg=''):
        '''If control.showFinal==True, print a message upon breakdown.'''
        tab=Tab()
        if self._control.showFinal:
            print('%s%s solve broke down: %s' % (tab, self.name(), msg))

    def reportFailure(self, iter, normR, normB):
        '''If control.showFinal==True, print a message upon failure to converge.'''
        tab=Tab()
        if self._control.showFinal:
            normRel = normR
            if normR != 0:
                normRel = normR/normB
            print('%s%s solve FAILED: iters=%7d, ||r||/r0=%12.5g' %
                (tab, self.name(), iter, normR/normB))
