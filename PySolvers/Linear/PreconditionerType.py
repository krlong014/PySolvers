from abc import ABC, abstractmethod
from . Preconditioner import IdentityPreconditioner

class PreconditionerType(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def form(self, A):
        pass

class IdentityPreconditionerType(PreconditionerType):

    def __init__(self):
        super().__init__()

    def form(self, A):
        return IdentityPreconditioner()
