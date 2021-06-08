from abc import ABC, abstractmethod

class Preconditioner(ABC):
    '''
    Preconditioner is an abstract base class for two-sided preconditioners.
    '''
    def __init__(self):
        pass

    @abstractmethod
    def applyLeft(self, vec):
        '''Apply the preconditioner from the left.'''
        pass

    @abstractmethod
    def applyRight(self, vec):
        '''Apply the preconditioner from the left.'''
        pass


class LeftPreconditioner(Preconditioner):
    '''Base class for left preconditioners.'''
    def __init__(self):
        super().__init__()

    def applyRight(self, vec):
        return vec



class RightPreconditioner(Preconditioner):
    '''Base class for right preconditioners.'''
    def __init__(self):
        super().__init__()

    def applyLeft(self, vec):
        return vec


class IdentityPreconditioner:
    def __init__(self):
        super().__init__()

    def applyLeft(self, vec):
        '''Apply the preconditioner from the left.'''
        return vec

    def applyRight(self, vec):
        '''Apply the preconditioner from the left.'''
        return vec
