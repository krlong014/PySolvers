from . Preconditioner import GenericPreconditioner
from . PreconditionerType import PreconditionerType
from . VCycleSolver import AMGVCycleSolver
from . ClassicSmoothers import GaussSeidelSmoother
from . IterativeLinearSolver import CommonSolverArgs


class AMG(PreconditionerType):
    def __init__(self, numIters=5, numLevels=2,
                 nuPre=2, nuPost=2, smoother=GaussSeidelSmoother):
        super().__init__()
        self.numIters = numIters
        self.numLevels = numLevels
        self.nuPre = nuPre
        self.nuPost = nuPost
        self.smoother = smoother

    def form(self, A):
        return AMGPreconditioner(A, numIters=self.numIters,
                                 numLevels=self.numLevels, nuPre=self.nuPre,
                                 nuPost=self.nuPost, smoother=self.smoother)





class AMGPreconditioner(GenericPreconditioner):
    '''
    Smoothed aggregation based multilevel preconditioner
    '''
    def __init__(self, A, numIters=5, numLevels=2,
                 nuPre=2, nuPost=2, smoother=GaussSeidelSmoother):
        '''
        AMG preconditioner constructor
        '''
        super().__init__()

        self._A = A
        control = CommonSolverArgs(maxiter=numIters, failOnMaxiter=False)
        self._solver = AMGVCycleSolver(control=control, numLevels=numLevels,
                                       nuPre=nuPre, nuPost=nuPost,
                                       smoother=smoother, name='AMG prec')
        self._solver.freezeMatrix()


    def apply(self, vec):
        '''Apply the multilevel operator to a vector.'''
        result = self._solver.solve(self._A, vec)
        if result.success():
            return result.soln()
        raise RuntimeError('ML preconditioner failed: {}'.format(result))
