from abc import ABC, abstractmethod
from .. NamedObject import NamedObject

# -----------------------------------------------------------------------------
# LinearSolverType

class LinearSolverType(ABC, NamedObject):
    def __init__(self, name=''):
        super().__init__(name=name)


    @abstractmethod
    def makeSolver(self, name=None):
        '''Creates a solver object with the specified arguments.'''
        pass


# -----------------------------------------------------------------------------
# LinearSolver

class LinearSolver(ABC, NamedObject):
    '''
    Class LinearSolver is a base class for linear solvers that
    provides a common interface for solving equations.
    '''
    def __init__(self, name=''):
        super().__init__(name=name)
        self._matrixFrozen = False

    @abstractmethod
    def solve(self, A, b):
        '''Solve A*x=b for x. Returns a SolveStatus object.'''
        pass

    def freezeMatrix(self):
        self._matrixFrozen = True

    def unfreezeMatrix(self):
        self._matrixFrozen = False

    def matrixFrozen(self):
        return self._matrixFrozen
