# ============================================================================
#
# Class IterativeLinearSolverType is a base class for factory objects that
# construct iterative solvers. This additional level of indirection is needed
# to allow construction of solvers deep in functions outside of the user's
# direct control.
#
# Class CommonSolverArgs collects typical control parameters for linear solvers.
# The user will normally set solver parameters through this object.
#
# Class IterativeLinearSolver is a base class for linear solvers that handles
# common maintenance jobs.
#
# Function mvmult() provides a common interface for matrix-vector multiplication
# usable by both numpy 2D arrays and scipy sparse matrices.
#
# Katharine Long, Texas Tech University, 2020-2021.
# ============================================================================


from abc import ABC, abstractmethod
import numpy as np
import scipy.sparse as sp
import numpy.linalg as npla
from PyTab import Tab
from . PreconditionerType import IdentityPreconditionerType
from . LinearSolver import LinearSolver, LinearSolverType

# -----------------------------------------------------------------------------
# CommonSolverArgs

class CommonSolverArgs:
    '''
    Class CommonSolverArgs ollects typical control parameters for linear
    solvers. The user will normally set solver parameters through this object.

    * Attributes:
        * maxiter -- maximum number of iterations allowed before stopping.
        * tau -- relative residual tolerance.
        * precond -- type of preconditioner to be used. This is a factory object
        that will build a preconditioner given a matrix.
        * norm -- the norm that will be used in checking for convergence
        * showIters -- whether to print out iteration status
        * interval -- number of iterations between output of status
        * showFinal -- whether to print convergence or error information upon
        termination
    '''
    def __init__(self,
                maxiter=100,
                tau=1.0e-8,
                precond=IdentityPreconditionerType(),
                norm=npla.norm,
                showIters=True,
                showFinal=True,
                interval=1):
        '''Constructor'''
        self.maxiter = maxiter
        self.tau = tau
        self.precondType = precond
        self.norm = norm
        self.showIters = showIters
        self.showFinal = showFinal
        self.interval = interval




# -----------------------------------------------------------------------------
# IterativeLinearSolverType

class IterativeLinearSolverType(LinearSolverType):
    '''
    Class IterativeLinearSolverType is a base class for factory objects that
    construct solvers. This additional level of indirection is sometimes
    needed to allow construction of solvers deep in functions outside of the
    user's direct control (for example, in a nonlinear solver or nested solver).
    '''
    def __init__(self, args=CommonSolverArgs(), name=''):
        '''Constructor.'''
        super().__init__(name)
        self._args = args

    def args(self):
        '''Return the arguments to be used in constructing the solver.'''
        return self._args


# -----------------------------------------------------------------------------
# IterativeLinearSolver

class IterativeLinearSolver(LinearSolver):
    '''
    Class IterativeLinearSolver is a base class for linear solvers that handles
    common parameters and common maintenance jobs.
    '''
    def __init__(self, args, name=''):
        super().__init__(name)
        self._args = args

    def maxiter(self):
        '''Return the maximum number of iterations allowed.'''
        return self._args.maxiter

    def tau(self):
        '''Return the relative residual tolerance.'''
        return self._args.tau

    def precond(self):
        '''Return the preconditioner factory for building preconditioners.'''
        return self._args.precondType

    def norm(self, x):
        '''Evaluate the norm of a vector x using the solver's specified norm.'''
        return self._args.norm(x)

    def iterOut(self, iter, normR, normR0):
        '''
        If args.showIters==True, print information about the current iteration
        at intervals specified by the args.interval parameter. Otherwise,
        do nothing.
        '''
        tab = Tab()
        if self._args.showIters and (iter % self._args.interval)==0:
            print('%s%s iter=%7d ||r||=%12.5g ||r||/r0=%12.5g' %
                  (tab, self.name(), iter, normR, normR/normR0))

    def reportSuccess(self, iter, normR, normR0):
        '''If args.showFinal==True, print a message upon convergence.'''
        if self._args.showFinal:
            print('%s solve succeeded: iters=%7d, ||r||/r0=%12.5g' %
                (self.name(), iter, normR/normR0))

    def reportBreakdown(self, msg=''):
        '''If args.showFinal==True, print a message upon breakdown.'''
        if self._args.showFinal:
            print('%s solve broke down: %s' % (self.name(), msg))

    def reportFailure(self, iter, normR, normR0):
        '''If args.showFinal==True, print a message upon failure to converge.'''
        if self._args.showFinal:
            print('%s solve FAILED: iters=%7d, ||r||/r0=%12.5g' %
                (self.name(), iter, normR/normR0))


# -----------------------------------------------------------------------------
# mvmult


def mvmult(A, x):
    '''
    Unified mvmult user interface for both scipy.sparse and numpy matrices.
    In scipy.sparse, mvmult is done using the overloaded * operator, e.g., A*x.
    In numpy, mvmult is done using the dot() function, e.g., dot(A,x).
    This function chooses which to use based on whether or not A is stored as
    a sparse matrix.
    '''

    if sp.issparse(A):
        return A*x
    else:
        return np.dot(A,x)
