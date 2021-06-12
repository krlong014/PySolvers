from abc import ABC, abstractmethod

# -----------------------------------------------------------------------------
# LinearSolverType

class LinearSolverType(ABC):
    def __init__(self, name=''):
        self._name = name


    @abstractmethod
    def makeSolver(self, name=None):
        '''Creates a solver object with the specified arguments.'''
        pass

    def name(self):
        '''
        Return a descriptive name for this solver. Useful with nested solvers.
        '''
        return self._name




# -----------------------------------------------------------------------------
# LinearSolver

class LinearSolver(ABC):
    '''
    Class LinearSolver is a base class for linear solvers that
    provides a common interface for solving equations.
    '''
    def __init__(self, name=''):
        self._name = name
        self._matrixFrozen = False
        self._precFrozen = False

    @abstractmethod
    def solve(self, A, b):
        '''Solve A*x=b for x. Returns a SolveStatus object.'''
        pass

    def name(self):
        '''
        Return a descriptive name for this solver. Useful with nested solvers.
        '''
        return self._name

    def freezeMatrix(self):
        self._matrixFrozen = True

    def unfreezeMatrix(self):
        self._matrixFrozen = False

    def matrixFrozen(self):
        return self._matrixFrozen

    def freezePrec(self):
        self._precFrozen = True

    def unfreezePrec(self):
        self._precFrozen = False

    def precFrozen(self):
        return self._precFrozen
