from abc import ABC, abstractmethod
import scipy.sparse


class MultilevelSequence(ABC):
    '''
    This class stores a sequence of update operators, downdate operators,
    and system matrices for use in multilevel solvers and preconditioning.
    Level 0 is the coarsest level; level numLevels-1 is the finest.

    1. I_up[k] maps level k to level k+1 (for k<N-1)
    2. I_down[k] maps level k+1 to level k (for k<N-1)
    3. A[k] is the system matrix at level k (for k=0 to N-1)
    '''

    def __init__(self, numLevels, normalize=True):
        '''
        Initialize the refinement sequence with empty operator arrays.
        '''
        self._numLevels = numLevels
        self._ops = [None]*numLevels
        self._updates = [None]*numLevels
        self._downdates = [None]*numLevels
        self._normalize = normalize

    @abstractmethod
    def makeProlongator(self, k):
        '''
        Form and return the prolongation operator to go from level k to level
        k+1. To be implemented by derived classes.
        '''
        pass

    def numLevels(self):
        '''Return the number of levels.'''
        return self._numLevels

    def update(self, k):
        '''Get the update operator mapping level k to level k+1.'''
        return self._updates[i]

    def downdate(self, k):
        '''Get the downdate operator mapping level k+1 to level k.'''
        return self._downdates[k]

    def matrix(self, k):
        return self._ops[k]

    def _setUpdate(self, k, I_up):
        self._updates[k] = I_up
        I_down = makeRestrictionOp(I_up, self.normalize)
        self._downdates[k] = I_down
        self._ops[k] = self._downdates[k]*(self._ops[k+1]*self._updates[k])

    def _setFineMatrix(self, A_fine):
        self._ops[self._numLevels-1] = A_fine


def makeRestrictionOp(I_up, normalize=True):
    '''
    Given a restriction operator I_down, create a prolongation operator I_up.
    The restriction operator will be the transpose of the prolongation
    operator, with optional row normalization.
    '''

    # Find the transpose
    I_down = I_up.transpose(copy=True).tolil()

    # Optionally normalize the rows
    if normalize:
        for r in range(I_down.shape[0]):
            row = I_down.getrowview(r)
            nrm = row.sum()
            row /= nrm

    # Convert to CSR before returning
    return I_down.tocsr()
