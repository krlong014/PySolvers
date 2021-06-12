from . MLHierarchy import MLHierarchy
from . SmoothedAggregation import SmoothedAggregationMLHierarchy
from .. SolveStatus import SolveStatus
from . ClassicSmoothers import GaussSeidelSmoother
from . VCycleManager import VCycleManager
from . IterativeLinearSolver import (IterativeLinearSolver,
                                     IterativeLinearSolverType,
                                     CommonSolverArgs)
from PyTab import Tab
import numpy.linalg as la
import scipy.sparse.linalg as spla
import numpy as np
import scipy.sparse as sp

class AMGVCycle(IterativeLinearSolverType):

    def __init__(self, control=CommonSolverArgs(), numLevels=2,
                 nuPre=2, nuPost=2, smoother=GaussSeidelSmoother,
                 name='AMGVCycle'):
        super().__init__(args=control)
        self.numLevels = numLevels
        self.nuPre = nuPre
        self.nuPost = nuPost
        self.smoother = smoother

    def makeSolver(self, name=None):
        '''Creates an AMG VCycle solver object with the specified parameters.'''
        useName = name
        if useName==None:
            useName = self.name()
        return AMGVCycleSolver(name=useName, control=self.args(),
                               numLevels = self.numLevels,
                               nuPre = self.nuPre,
                               nuPost = self.nuPost,
                               smoother = self.smoother
                               )




class AMGVCycleSolver(IterativeLinearSolver):
    def __init__(self, control=CommonSolverArgs(),
                 numLevels=2, nuPre=2, nuPost=2,
                 smoother=GaussSeidelSmoother, name='AMGVCycle'):
        super().__init__(args=control, name=name)
        self.numLevels = numLevels
        self.nuPre = nuPre
        self.nuPost = nuPost
        self.smoother = smoother
        self._cycleMgr = None

    def solve(self, A, b):
        tab = Tab()

        # Get size of matrix
        n,nc = A.shape
        # Make sure matrix is square
        assert(n==nc)
        # Make sure A and b are compatible
        assert(n==len(b))

        # Check for the trivial case b=0, x=0
        normB = self.norm(b)
        if normB == 0.0:
            return self.handleConvergence(0, np.zeros_like(b), 0, 0)

        # Create vectors for residual r and solution x
        r = np.copy(b)
        x = np.copy(b)

        if self._cycleMgr==None or not self.matrixFrozen():
            mlh = SmoothedAggregationMLHierarchy(A, numLevels=self.numLevels)

            self._cycleMgr = VCycleManager(mlh, nuPre=self.nuPre,
                                           nuPost=self.nuPost,
                                           smoother=self.smoother)

        # Main loop
        for k in range(self.maxiter()):
            # Run a V-cycle
            x = self._cycleMgr.runCycle(b, x)

            # Compute residual
            r = b - A*x

            # Check for convergence
            normR = self.norm(r)

            self.reportIter(k, normR, normB)
            if normR < self.tau()*normB:
                return self.handleConvergence(k, x, normR, normB)

        # If we're here, maxiter has been reached. This is normally a failure,
        # but may be acceptable if failOnMaxiter is set to false.
        return self.handleMaxiter(k, x, normR, normB)
