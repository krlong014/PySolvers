# ============================================================================
# Preconditioned conjugate gradient solver and factory class.
#
# Class PGCSolver is a subtype of iterative linear solver
#
# Katharine Long, Texas Tech University, 2020-2021.
# ============================================================================

import numpy.linalg as npla
import numpy as np
from . PreconditionerType import IdentityPreconditionerType
from . IterativeLinearSolver import (IterativeLinearSolver, mvmult,
                                      CommonSolverArgs,
                                      IterativeLinearSolverType)
from .. SolveStatus import SolveStatus


# -----------------------------------------------------------------------------
# PCG factory class

class PCG(IterativeLinearSolverType):
    def __init__(self, args=CommonSolverArgs(), name='PCG'):
        super().__init__(args=args, name=name)

    def makeSolver(self, name=None):
        '''Creates a PCG solver object with the specified parameters.'''
        useName = name
        if useName==None:
            useName = self.name()
        return PCGSolver(self.args(), useName)


# -----------------------------------------------------------------------------
# PCGSolver solver class

class PCGSolver(IterativeLinearSolver):
    '''
    Preconditioned conjugate gradient iteration for solving Ax=b with A SPD
    * A -- system matrix, can be numpy 2D array or scipy sparse matrix. Must be
        SPD for the algorithm to succeed, though this is not checked.
    * b -- numpy vector, must be a numpy array
    * maxiter -- maximum number of iterations
    * tau -- stopping tolerance tau. Success will be declared when the relative
        residual is less than tau.
    * monitor --
    '''
    def __init__(self, args=CommonSolverArgs(), name='PCG'):
        '''Constructor'''

        super().__init__(args=args, name=name)

        # Done PCGSolver constructor

    def solve(self, A, b ):
        '''
        Solve the system A*x=b for x.
        * Input:
            * A -- System matrix, can be a numpy 2D array or a scipy sparse matrix.
                 Must be SPD; this is not checked (doing so is too expensive).
            * b -- RHS vector, must be a numpy 1D array compatible with A.
        * Return:
            * A SolveStatus object containing the solution estimate, a
              success/failure flag, and convergence information.
        '''

        # Get size of matrix
        n,nc = A.shape
        # Make sure matrix is square
        assert(n==nc)
        # Make sure A and b are compatible
        assert(n==len(b))

        # Check for the trivial case b=0, x=0
        normB = self.norm(b)
        if normB == 0.0:
            self.reportSuccess(0, 0, normB)
            return SolveStatus(conv=True, soln=np.zeros_like(b),
                               resid=0, iters=0)

        # Form the preconditioner
        precond = self.precond().form(A)

        # Initialize the step, residual, and solution vectors
        r = 1.0*b
        p = precond.applyRight(r)
        u = 1.0*p
        x = np.zeros_like(b)

        # We'll use ||b|| for computing relative residuals
        normB = self.norm(b)


        uDotR = np.dot(u,r) # Should never be zero, since we've caught b=0
        # Check anyway
        if uDotR==0.0:
            self.reportBreakdown(msg='dot(u,r)==0')
            return SolveStatus(success=False, msg='breakdown dot(u,r)==0')


        # Preconditioned CG loop
        for k in range(self.maxiter()):
            # Compute A*p
            Ap = mvmult(A,p)

            pTAp = np.dot(p,Ap)
            if pTAp==0.0:
                self.reportBreakdown(prefix='PCG', msg='breakdown dot(p, Ap)==0')
                return SolveStatus(success=False, soln=x, resid=-1, iters=k,
                                    msg='breakdown dot(p, Ap)==0')

            # calculate step length
            alpha = uDotR/pTAp

            x = x + alpha*p
            r = r - alpha*Ap
            u = precond.applyRight(r)

            normR = self.norm(r)
            self.iterOut(k, normR, normB)

            if normR <= self.tau()*normB:
                self.reportSuccess(k+1, normR, normB)
                return SolveStatus(success=True, iters=k+1, soln=x, resid=normR)

            newUDotR = np.dot(u,r)
            beta = newUDotR/uDotR
            uDotR = newUDotR

            p = u + beta*p

        # Failure to converge.
        self.reportFailure(k+1, normR, normB)
        return SolveStatus(success=False, iters=k+1, soln=x, resid=normR,
                           msg='failure to converge')

        # Done PCG solve()
