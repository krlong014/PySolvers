# ============================================================================
# Preconditioned conjugate gradient solver and factory class.
#
# Class PCG is a factory class that builds a PCGSolver using the makeSolver()
# function in the IterativeLinearSolverType API.
#
# Class PGCSolver is a solver object.
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
            return self.handleConvergence(0, np.zeros_like(b), 0, 0)

        # Form the preconditioner
        precond = self.precond().form(A)

        # Initialize the step, residual, and solution vectors
        r = np.copy(b)
        p = precond.applyRight(r)
        u = np.copy(p)
        x = np.zeros_like(b)

        uDotR = np.dot(u,r) # Should never be zero, since we've caught b=0
        # Check anyway
        if uDotR==0.0:
            return self.handleBreakdown(0, 'breakdown dot(u,r)==0')


        # Preconditioned CG loop
        for k in range(self.maxiter()):
            # Compute A*p
            Ap = mvmult(A,p)

            pTAp = np.dot(p,Ap)
            if pTAp==0.0:
                return self.handleBreakdown(k, 'breakdown dot(p, Ap)==0')

            # calculate step length
            alpha = uDotR/pTAp

            # Make step and compute updated residual
            x = x + alpha*p
            r = r - alpha*Ap
            u = precond.applyRight(r)

            normR = self.norm(r)
            self.reportIter(k, normR, normB)

            # Check for convergence
            if ((normR <= self.tau()*normB) or
                    ((not self.failOnMaxiter()) and k==self.maxiter()-1)):
                return self.handleConvergence(k, x, normR, normB)

            # Find next step direction
            newUDotR = np.dot(u,r)
            beta = newUDotR/uDotR
            uDotR = newUDotR

            p = u + beta*p

        # If we're here, maxiter has been reached. This is normally a failure,
        # but may be acceptable if failOnMaxiter is set to false.
        return self.handleMaxiter(k, x, normR, normB)


        # Done PCG solve()
