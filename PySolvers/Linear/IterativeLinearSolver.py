# ============================================================================
#
# Class IterativeLinearSolverType is a base class for factory objects that
# construct iterative solvers. This additional level of indirection is needed
# to allow construction of solvers deep in functions outside of the user's
# direct control.
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
from .. SolveStatus import SolveStatus
from .. IterativeSolver import IterativeSolver, CommonSolverArgs


# -----------------------------------------------------------------------------
# IterativeLinearSolverType

class IterativeLinearSolverType(LinearSolverType):
    '''
    Class IterativeLinearSolverType is a base class for factory objects that
    construct solvers. This additional level of indirection is sometimes
    needed to allow construction of solvers deep in functions outside of the
    user's direct control (for example, in a nonlinear solver or nested solver).
    '''
    def __init__(self,
                 control=CommonSolverArgs(),
                 precond=IdentityPreconditionerType(),
                 name=''):
        '''Constructor.'''
        super().__init__(name)
        self._control = control
        self._precondType = precond

    def precond(self):
        '''Return the preconditioner factory for building preconditioners.'''
        return self._precondType

    def control(self):
        '''Return the arguments to be used in constructing the solver.'''
        return self._control


# -----------------------------------------------------------------------------
# IterativeLinearSolver

class IterativeLinearSolver(LinearSolver, IterativeSolver):
    '''
    Class IterativeLinearSolver is a base class for linear solvers that handles
    common parameters and common maintenance jobs.
    '''
    def __init__(self, control, precond=IdentityPreconditionerType(), name=''):
        super(LinearSolver, self).__init__(control=control, name=name)
        super(IterativeSolver, self).__init__(name=name)
        self._precondType = precond
        self._precFrozen = False

    def precondType(self):
        '''Return the preconditioner factory for building preconditioners.'''
        return self._precondType


    def setTolerance(self, tau):
        super().setTolerance(tau)

    def freezePrec(self):
        self._precFrozen = True

    def unfreezePrec(self):
        self._precFrozen = False

    def precFrozen(self):
        return self._precFrozen



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
