# ============================================================================
# Preconditioned GMRES solver and factory class.
#
# Class GMRES is a factory class that builds a GMRESSolver using the
# makeSolver() function in the IterativeLinearSolverType API.
#
# Class GMRESSolver is a solver object.
#
# Katharine Long, Texas Tech University, 2020-2021.
# ============================================================================

from copy import deepcopy
import numpy as np
import numpy.linalg as npla
import scipy.sparse as sp
from . Givens import findGivensCoefficients, applyGivens, applyGivensInPlace
from . PreconditionerType import IdentityPreconditionerType
from . IterativeLinearSolver import (IterativeLinearSolver, mvmult,
                                      IterativeLinearSolverType)
from .. IterativeSolver import (IterativeSolver, CommonSolverArgs)
from .. SolveStatus import SolveStatus


# -----------------------------------------------------------------------------
# GMRES factory class

class GMRES(IterativeLinearSolverType):
    def __init__(self,
                 control=CommonSolverArgs(),
                 precond=IdentityPreconditionerType(),
                 name='GMRES'):
        super().__init__(control=control, precond=precond, name=name)

    def makeSolver(self, name=None):
        '''Creates a GMRES solver object with the specified parameters.'''
        useName = name
        if useName==None:
            useName = self.name()
        return GMRESSolver(self.control(), precond=self.precond(),
                           name=useName)


# -----------------------------------------------------------------------------
# GMRES solver class

class GMRESSolver(IterativeLinearSolver):
    def __init__(self,
                 control=CommonSolverArgs(),
                 precond=IdentityPreconditionerType(),
                 name='GMRES'):
        '''Constructor'''
        super().__init__(control=control, precond=precond, name=name)


    def solve(self, A, b):

        # Get shape of matrix and check that it's square. We'll need the size n
        # to set the dimension of the Arnoldi vectors (i.e., the number of rows in
        # the matrix Q).
        n,nc = A.shape
        assert(n==nc)
        # Make sure A and b are compatible
        assert(n==len(b))

        # Check for the trivial case b=0, x=0
        norm_b = self.norm(b)
        if norm_b == 0.0:
            return self.handleConvergence(0, np.zeros_like(b), 0, 0)

        # Form the preconditioner
        if self.precond == None or not self.precFrozen():
            precond = self.precondType().form(A)

        # Allocate space for Arnoldi results
        maxiters = self.maxiter()
        # Q is n by m+1 after m Arnoldi steps, preallocate to maxiters+1 columns
        Q = np.zeros([n, maxiters+1])
        # HBar is m+1 by m after m Arnoldi steps, preallocated to m=maxiters.
        # We will triangularize HBar via Givens as we go.
        HBar = np.zeros([maxiters+1, maxiters])

        # Create an array in which we'll store all the Givens cosines and sines
        CS = np.zeros([maxiters,2])


        # Initial residual is b.
        r0 = b

        # Initialize q_0 and beta
        beta = npla.norm(r0)
        Q[:,0] = r0 / beta

        # Initialize RHS for least squares problem.
        # Least squares problem is to minimize ||HBar y - beta e1||.
        e1 = np.zeros(maxiters+1)
        e1[0] = 1.0
        g = beta*e1 # Will be modified by Givens rotations as we go

        # Flag to indicate whether Arnoldi algorithm has hit breakdown
        # (In Arnoldi's algorithm, breakdown is a good thing!)
        arnoldiBreakdown = False

        # Outer Arnoldi loop for up to maxiters vectors
        for k in range(maxiters):

            # Form A*M_R^-1*q_k
            u = mvmult(A, precond.applyRight(Q[:,k]))

            # Inner modified Gram-Schmidt loop
            for j in range(k+1):
                HBar[j,k]=np.dot(Q[:,j], u)
                u -= HBar[j,k]*Q[:,j]

            # Fill in the extra entry in HBar
            HBar[k+1,k]=npla.norm(u)

            # Check for breakdown of Arnoldi. Recall that Arnoldi breaks down
            # iff the iteration count is equal to the degree of the minimal
            # polynomial of A. Therefore, the exact solution is in the current
            # Krylov space and we have converged.
            hLastColNorm = npla.norm(HBar[0:k+1,k])
            if abs(HBar[k+1,k]) <= 1.0e-16 * hLastColNorm:
                arnoldiBreakdown = True
            else:
                Q[:,k+1]=u/HBar[k+1,k]

            # We've now updated the Hessenberg matrix HBar with the
            # most recent column. The next step is to triangularize
            # it with Givens rotations.

            # First, apply all previous Givens rotations to
            # it, in order.
            for j in range(k):
                #HBar[:,k] = applyGivensInPlace(HBar[:,k], CS[j,0], CS[j,1], j)
                applyGivensInPlace(HBar[:,k], CS[j,0], CS[j,1], j)


            # Find the Givens rotation that will zero
            # out the bottom entry in the last column.
            CS[k,:]=findGivensCoefficients(HBar[:,k], k)

            # Apply the Givens rotation to kill the subdiagonal in the most
            # recent column
            #HBar[:,k] = applyGivens(HBar[:,k], CS[k,0], CS[k,1], k)
            applyGivensInPlace(HBar[:,k], CS[k,0], CS[k,1], k)
            # Apply the same rotation to the RHS of the least squares problem
            #g = applyGivens(g, CS[k,0], CS[k,1], k)
            applyGivensInPlace(g, CS[k,0], CS[k,1], k)

            # The current residual norm is the absolute value of the final entry in
            # the RHS vector g.
            norm_r_k = np.abs(g[k+1])

            # Print the current residual
            self.reportIter(k, norm_r_k, norm_b)

            # Check for convergence
            if (arnoldiBreakdown==True) or (norm_r_k <= self.tau()*norm_b):
                y = npla.solve(HBar[0:k+1,0:k+1], g[0:k+1])
                x = precond.applyRight(np.dot(Q[:,0:k+1],y))
                # Compute residual, and compare to implicitly computed
                # residual
                resid = b - mvmult(A,x)
                norm_r_true = self.norm(resid)
                if norm_r_true <= self.tau()*norm_b:
                    return self.handleConvergence(k, x, norm_r_true, norm_b)
                else:
                    return SolveStatus(success=False, iters=k+1, soln=x,
                                       resid=norm_r_true,
                            msg=
                            '''
                            GMRES failure: true residual %12.5g did not meet tolerance
                            tau=%12.5g. Recursive residual was %12.5g.'''.
                            format(norm_r_true, self.tau(), norm_r_k))



        # If we're here, maxiter has been reached. This is normally a failure,
        # but may be acceptable if failOnMaxiter is set to false.
        return self.handleMaxiter(k, 0, norm_k, norm_b)
