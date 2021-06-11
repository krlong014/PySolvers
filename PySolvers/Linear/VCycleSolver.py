from . MultilevelSequence import MultilevelSequence
from . SolverState import SolverState
from Tab import *
from Debug import *
from ClassicSmoothers import GaussSeidelSmoother
import numpy.linalg as la
import scipy.sparse.linalg as spla
import numpy as np
import scipy.sparse as sp

class AMGSolver:
    def __init__(self, numLevels=2, nuPre=2, nuPost=2,
                 maxIters=100, tol=1.0e-8,
                 smoother=GaussSeidelSmoother):

    seq = SmoothedAggregationMLSequence(A, numLevels=numLevels)

    self._cycleMgr = VCycleManager(seq, nuPre=nuPre, nuPost=nuPost, smoother)

    self._maxIters = maxIters
    self._tol = tol


    def solve(self, A, b):

        # We'll use ||b|| for convergence testing
        bNorm = la.norm(b)

        # Create vectors for residual r and solution x
        r = np.copy(b)
        x = np.copy(b)

        # Short-circuit if b=0. Solution is easy.
        if bNorm == 0.0:
            return SolverState(True, soln=x, resid=0, iters=0, msg='zero RHS')


        # Main loop
        for k in range(self._maxIters):
            # Run a V-cycle
            x = self.runCycle(b, x)

            # Compute residual
            r = b - A*x

            # Check for convergence
            rNorm = la.norm(r)

            Debug.msg2(verb, tab1, 'Iter=%d relative resid=%g' % (k, rNorm/bNorm))
            if rNorm < self._tol*bNorm:
                return SolverState(True, soln=x, resid=rNorm, iters=k)

        # If we're here, the method didn't converge withing the maximum number
        # of iterations
        return SolverState(False, soln=x, resid=rNorm, iters=k)
