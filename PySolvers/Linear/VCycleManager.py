from . MultilevelSequence import MultilevelSequence
from . ClassicSmoothers import JacobiSmoother, GaussSeidelSmoother

import numpy.linalg as la
import scipy.sparse.linalg as spla
import numpy as np
import scipy.sparse as sp

class VCycleManager:
    # Constructor
    def __init__(self, mlSeq, nuPre=2, nuPost=2, smoother=GaussSeidelSmoother):
        self._numLevels = mlSeq.numLevels()
        self._mlSeq = mlSeq
        self._nuPre = nuPre
        self._nuPost = nuPost

        # Make sure the mlSeq argument is a MultilevelSequence
        assert(isinstance(mlSeq, MultilevelSequence))

        # Construct the smoothers at each level
        self._smoothers = self._numLevels * [None]
        for lev in range(self._numLevels):
            self._smoothers[lev] = smoother(self.mlSeq.matrix(lev))

    # Run a single V-cycle
    def runCycle(self, b, x):
        return self.runLevel(b, x, self._numLevels-1)


    # Carry out a V-cycle from level lev
    def runLevel(self, fh, xh, lev):

      # if at coarsest level, do a direct solve
      if lev==0:
        A_c = self._mlSeq.matrix(0)
        xOut = spla.spsolve(A_c, fh) # use SuperLU solver in scipy
        return xOut

      # Otherwise: pre-smooth, apply recursively, and post-smooth

      # Pre-smooth
      xh = self._smoothers[lev].apply(fh, xh, self._nuPre)

      # Find the residual after smoothing
      rh = fh - self._mlSeq.matrix(lev)*xh

      # Coarsen the residual
      r2h = self._mlSeq.downdate(lev-1)*rh

      # Recursively apply ML to solve A^{2h} e^{2h} = r^{2h}
      x2h = np.zeros_like(r2h)
      x2h = self.runLevel(r2h, x2h, lev-1)

      # Correct the solution by adding in the prolongation of the coarse-grid error
      xh = xh + self._mlSeq.update(lev-1)*x2h

      # Post-smooth to remove any high-frequency errors resulting from fine-grid
      # correction

      xh = self._smoothers[lev].apply(fh, xh, self._nuPost)

      return xh
