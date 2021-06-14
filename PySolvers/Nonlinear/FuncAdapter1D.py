import numpy as np
from abc import ABC, abstractmethod

class FuncAdapter1D(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def _evalF(self, x):
        pass

    @abstractmethod
    def _evalJ(self, x):
        pass

    def evalF(self, x):
        y = x[0]
        f = self._evalF(y)
        return np.array([f,])

    def evalJ(self, x):
        y = x[0]
        J = self._evalJ(y)
        return J*np.eye(1)
