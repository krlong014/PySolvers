from abc import ABC, abstractmethod
from PyTab import Tab

class LineSearch(ABC):

    def __init__(self, maxsteps=15, low=0.1, alpha=0.0001, report=True):
        self._maxsteps = maxsteps
        self._alpha = alpha
        self._report = report
        self._low = low
        self._norm = None

    @abstractmethod
    def search(self, x0, resid, newtStep, func):
        pass

    def maxsteps(self):
        return self._maxsteps

    def alpha(self):
        return self._alpha

    def low(self):
        return self._low

    def setNorm(self, norm):
        self._norm = norm

    def norm(self, x):
        if self._norm == None:
            raise RuntimeError('Norm not set in line search')
        return self._norm(x)

    def report(self, k, t, ratio):
        tab = Tab()
        if self._report:
            print('%sk=%4d t=%12.5g ||F_k||/||F_0||=%12.5g' % (
                tab, k, t, ratio))

class TrivialLinesearch(LineSearch):
    '''
    Trivial line search the immediately accepts the full Newton step. For
    testing only.
    '''
    def __init__(self, report=True):
        super().__init__(report=report)

    def search(self, x0, normF0, newtStep, func):
        x1 = x0 + newtStep
        F1 = func.eval(x1)
        normF1 = self.norm(F1)
        return (True, x1, F1, normF1)


class SimpleBacktrack(LineSearch):
    '''
    Implementation of the backtracking algorithm from Dennis & Schnabel 1996.
    '''
    def __init__(self, maxsteps=10, low=0.1, alpha=0.0001, report=True):
        super().__init__(maxsteps=maxsteps, low=low, alpha=alpha, report=report)

    def search(self, x0, normF0, newtStep, func):
        tab = Tab()
        t = 1.0
        for k in range(self.maxsteps()):
            x_k = x0 + t * newtStep
            F_k = func.evalF(x_k)
            normF_k = self.norm(F_k)
            ratio = normF_k/normF0
            self.report(k, t, ratio)
            # Test for convergence of line search
            if normF_k <= (1.0 - self.alpha()*t)*normF0:
                return (True, x_k, F_k, normF_k)
            # Shrink step
            factor = 0.5/ratio
            if factor<self.low():
                factor = self.low()
            t = t * factor

        # If here, the line step hasn't produced sufficient decrease.
        return (False, x_k, F_k, normF_k)
