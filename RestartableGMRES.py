import time
from copy import deepcopy
import scipy.linalg as la
import numpy as np
import scipy.sparse.linalg as spla
import scipy.sparse as sp
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
from Givens import findGivensCoefficients, applyGivens
from scipy.io import mmread
from BasicPreconditioner import *

# Unified mvmult user interface for both scipy.sparse and numpy matrices.
def mvmult(A, x):
    if sp.issparse(A):
        return A*x
    else:
        return np.dot(A,x)

# This is function applies GMRES to solve Ax=b for x with optional right
# preconditioning and optional restarts
# Input arguments:
# (*) A -- the system matrix in dense numpy form (no point in going sparse yet)
# (*) b -- the RHS vector as a numpy array
# (*) maxiters -- maximum number of iterations to attempt
# (*) tol -- relative residual tolerance
# (*) precond -- preconditioner
# (*) krylov_size -- maximum size of Krylov space. If this is reached, restart
#                    using the current solution estimate as an initial guess.
#
def GMRES(A, b, maxiters=100, tol=1.0e-6,
    precond=PreconditionerBase(), krylov_size=None):

    # We'll scale residual norms relative to ||b||
    norm_b = la.norm(b)

    # Use initial guess x0=0
    x0 = 0.0*b

    # Get shape
    n,nc = A.shape
    if n!=nc:
        raise RuntimeError('InnerGMRES: Non-square matrix; size is %d by %d'
            % (n,nc))

    # If not specified, set maximum Krylov subspace size to maxiters
    if krylov_size==None:
        krylov_size = maxiters

    # krylov_size > maxiters makes no sense, so set it to maxiters if
    # this happens
    if krylov_size > maxiters:
        krylov_size = maxiters

    # Allocate space for Arnoldi results
    # Q is n by m+1 after m Arnoldi steps, preallocate to maxiters+1 columns
    Q = np.zeros([n, krylov_size+1])
    # HBar is m+1 by m after m Arnoldi steps, preallocated to m=maxiters.
    # We will triangularize HBar via Givens as we go.
    HBar = np.zeros([krylov_size+1, krylov_size])

    # Create an array in which we'll store all the Givens cosines and sines
    CS = np.zeros([krylov_size,2])


    # Main loop
    iters = 0
    while True:
        # Re-zero Q, H, and CS
        Q = Q*0.0
        HBar = HBar*0.0
        CS = CS*0.0

        # Compute true residual at current solution estimate
        r0 = b - mvmult(A,x0)

        # Initialize q_0 and beta
        beta = la.norm(r0)
        Q[:,0] = r0 / beta

        # Initialize RHS for least squares problem.
        # Least squares problem is to minimize ||HBar y - beta e1||.
        e1 = np.zeros(krylov_size+1)
        e1[0] = 1.0
        g = beta*e1 # Will be modified by Givens rotations as we go

        # Flag to indicate whether Arnoldi algorithm has hit breakdown
        # (which is a good thing!)
        arnoldiBreakdown = False

        # Outer Arnoldi loop for up to krylov_size vectors
        for k in range(krylov_size):
            # Keep track of total iterations
            iters += 1

            # Form A*M_R^-1*q_k
            u = mvmult(A, precond.applyRight(Q[:,k]))

            # Inner modified Gram-Schmidt loop
            for j in range(k+1):
                HBar[j,k]=np.dot(Q[:,j], u)
                u -= HBar[j,k]*Q[:,j]

            # Fill in the extra entry in HBar
            HBar[k+1,k]=la.norm(u)

            # Check for breakdown of Arnoldi. Recall that Arnoldi breaks down
            # iff the iteration count is equal to the degree of the minimal
            # polynomial of A. Therefore, the exact solution is in the current
            # Krylov space and we have converged.
            hLastColNorm = la.norm(HBar[0:k+1,k])
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
                HBar[:,k] = applyGivens(HBar[:,k], CS[j,0], CS[j,1], j)


            # Find the Givens rotation that will zero
            # out the bottom entry in the last column.
            CS[k,:]=findGivensCoefficients(HBar[:,k], k)

            # Apply the Givens rotation to kill the subdiagonal in the most
            # recent column
            HBar[:,k] = applyGivens(HBar[:,k], CS[k,0], CS[k,1], k)
            # Apply the same rotation to the RHS of the least squares problem
            g = applyGivens(g, CS[k,0], CS[k,1], k)

            # The current residual norm is the absolute value of the final entry in
            # the RHS vector g.
            norm_r_k = np.abs(g[k+1])

            # Print the current residual
            print('\titer %4d\t%4d\tr=%12.5g' %(iters, k, (norm_r_k/norm_b)))

            # Check for convergence
            if (arnoldiBreakdown==True) or (norm_r_k <= tol*norm_b):
                print('GMRES converged!')
                y = la.solve(HBar[0:k+1,0:k+1], g[0:k+1])
                x = x0 + precond.applyRight(np.dot(Q[:,0:k+1],y))
                # Compute residual, and compare to implicitly computed
                # residual
                resid = b - mvmult(A,x)
                print('Implicit residual=%12.5g, true residual=%12.5g'
                    % (norm_r_k/norm_b, la.norm(resid)/norm_b))
                return (True, x)

            # Check for reaching maxiters without convergence
            if iters==maxiters:
                print('GMRES failed to converge after %g iterations'
                    % maxiters)
                return (False, 0)

        # If we can still iterate, then update current guess and try again
        y = la.solve(HBar[0:krylov_size,0:krylov_size], g[0:krylov_size])
        x0 = x0 + precond.applyRight(mvmult(Q[:,0:krylov_size],y))

        # On to next restart

    # This should never be reached: the main loop should be terminated by
    # convergence or by too many iterations
    return (False, 0)



# ---- Test program --------

if __name__=='__main__':

    rs = RandomState(MT19937(SeedSequence(123456789)))

    level = 20
    A = mmread('DH-Matrix-%d.mtx' % level)
    A = A.tocsr()
    n,nc = A.shape
    print('System is %d by %d' %(n,nc))

    if n < 12000:
        Adense = A.todense()
        print('\nCondition number ', np.linalg.cond(Adense))


    # Create a solution

    xEx = rs.rand(n)
    # Multiply the solution by A to create a RHS vector
    b = mvmult(A, xEx)

    tStart = time.time()

    drop = 1.0e-4
    ILU = ILURightPreconditioner(A, drop_tol=drop)


    # Run GMRES
    (conv,x) = GMRES(A,b,maxiters=500, tol=1.0e-6, precond=ILU)

    tFin = time.time()
    gmresTime = tFin - tStart

    # Print the error
    if conv:
        err = la.norm(x - xEx)/la.norm(xEx)
        print('\nRelative error norm = %10.3g' % err)
    else:
        print('GMRES failed')

    print('\nGMRES wall clock time %10.3g seconds' % gmresTime)

    # For comparison, do a sparse direct solve using SuperLU
    tStart = time.time()
    LU = spla.splu(A.tocsc())
    xDirect = LU.solve(b)
    tFin = time.time()

    print('\nSparse LU wall clock time %10.3g seconds' % (tFin-tStart))
