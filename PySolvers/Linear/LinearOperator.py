from abc import ABC, abstractmethod
from . LinearSolverType import LinearSolverType

class LinearOperator(ABC):

    def __init__(self, name):
        self._name = name

    @abstractmethod
    def apply(self, vec):
        pass

    def inverse(self, solverType):
        return InverseOp(self, solverType)

    def name(self):
        return self._name

    def __str__(self):
        if self._name == None:
            return 'LinearOperator'
        else:
            return self._name

    def __neg__(self):
        return ScalarMultOp(self, -1)

    def __mul__(self, other):
        if isinstance(other, Number):
            return ScalarMultOp(self, other)
        elif isinstance(other, np.ndarray):
            return self.apply(other)
        elif isinstance(other, LinearOperator):
            return ComposedOp(self, other)
        else:
            raise ValueError('unable to interpret product {} times {}'.
                             format(self, other))

    def __rmul__(self, other):
        if isinstance(other, Number):
            return ScalarMultOp(self, other)
        else:
            raise ValueError('unable to interpret rmul product {} times {}'.
                             format(other, self))

    def __truediv__(self, other):
        if isinstance(other, Number):
            return ScalarMultOp(self, 1/other)
        else:
            raise ValueError('unable to interpret quotient {} div by {}'.
                             format(self, other))

    def __add__(self, other):
        if isinstance(other, LinearOperator):
            return SumOp(self, other)
        else:
            raise ValueError('unable to interpret sum {} plus {}'.
                             format(self, other))

    def __sub__(self, other):
        if isinstance(other, LinearOperator):
            return SumOp(self, -other)
        else:
            raise ValueError('unable to interpret subtraction {} minus {}'.
                             format(self, other))


class ComposedOp(LinearOperator):

    def __init__(self, A, B):
        super().__init__('{}*{}'.format(A.name(), B.name()))
        self._A = A
        self._B = B

    def apply(self, vec):
        tmp = self._B.apply(vec)
        rtn = self._A.apply(tmp)
        return rtn

class ScalarMultOp(LinearOperator):

    def __init__(self, A, alpha):
        super().__init__('{}*{}'.format(alpha, A.name()))
        self._A = A
        self._alpha = alpha

    def apply(self, vec):
        tmp = self._A.apply(vec)
        rtn = self._alpha * tmp
        return rtn

class SumOp(LinearOperator):

    def __init__(self, A, B):
        super().__init__('({}+{})'.format(A.name(), B.name()))
        self._A = A
        self._B = B

    def apply(self, vec):
        tmp1 = self._A.apply(vec)
        tmp2 = self._B.apply(tmp)
        return tmp1 + tmp2


class InverseOp(LinearOperator):

    def __init__(self, A, solverType):
        super().__init__('({})^{-1}'.format(A))
        self._A = A
        self._solver = solverType.makeSolver()

    def apply(self, vec):
        result = self._solver.solve(self._A, vec)
        if result.success():
            return result.soln
        else:
            raise RuntimeError(
                '''Solve failure during application of inverse operator {}:
                solve status was: {}'''.format(self.name(), result))



class IdentityOp(LinearOperator):

    def __init__(self, A, solverType):
        super().__init__('I')

    def apply(self, vec):
        return vec
