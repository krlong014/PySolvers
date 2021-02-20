# Preconditioned GMRES solver for Math 5344, Fall 2020.
# Katharine Long, Texas Tech.
# This code is in the public domain.

import time
from copy import deepcopy
import scipy.linalg as la
import numpy as np
import scipy.sparse.linalg as spla
import scipy.sparse as sp
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
from Givens import findGivensCoefficients, applyGivens, applyGivensInPlace
from scipy.io import mmread
from BasicPreconditioner import *

# A simple timer class
class MyTimer:
    def __init__(self, name):
        self.name = name
        self.start = time.time()

    def stop(self):
        self.stop = time.time()

    def walltime(self):
        return self.stop - self.start

# Unified mvmult user interface for both scipy.sparse and numpy matrices.
# In scipy.sparse, mvmult is done using the overloaded * operator, e.g., A*x.
# In numpy, mvmult is done using the dot() function, e.g., dot(A,x).
# This function chooses which to use based on whether A is stored as
# a sparse matrix.
def mvmult(A, x):
    if sp.issparse(A):
        return A*x
    else:
        return np.dot(A,x)

# This is function applies GMRES to solve Ax=b for x with optional right
# preconditioning
# Input arguments:
# (*) A -- the system matrix in dense numpy form (no point in going sparse yet)
# (*) b -- the RHS vector as a numpy array
# (*) maxiters -- maximum number of iterations to attempt
# (*) tol -- relative residual tolerance. If ||r_k|| <= tol*||b||, stop.
# (*) precond -- preconditioner. Default is a do-nothing preconditioner. The
#                preconditioner needs to be a class providing an applyRight()
#                function that carries out the operation $$M_R^{-1} v$$ on a
#                vector v, resulting the result as a numpy array.
#
def GMRES(A, b, maxiters=100, tol=1.0e-6,
    precond=PreconditionerBase()):

    # We'll scale residual norms relative to ||b||
    norm_b = la.norm(b)

    # Get shape of matrix and check that it's square. We'll need the size n
    # to set the dimension of the Arnoldi vectors (i.e., the number of rows in
    # the matrix Q).
    n,nc = A.shape
    if n!=nc:
        raise RuntimeError('InnerGMRES: Non-square matrix; size is %d by %d'
            % (n,nc))


    # Allocate space for Arnoldi results
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
    beta = la.norm(r0)
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
        print('\titer %4d\tr=%12.5g' %(k, (norm_r_k/norm_b)))

        # Check for convergence
        if (arnoldiBreakdown==True) or (norm_r_k <= tol*norm_b):
            print('GMRES converged!')
            y = la.solve(HBar[0:k+1,0:k+1], g[0:k+1])
            x = precond.applyRight(np.dot(Q[:,0:k+1],y))
            # Compute residual, and compare to implicitly computed
            # residual
            resid = b - mvmult(A,x)
            print('Implicit residual=%12.5g, true residual=%12.5g'
                % (norm_r_k/norm_b, la.norm(resid)/norm_b))
            return (True, k+1, x)

    # Check for reaching maxiters without convergence
    print('GMRES failed to converge after %g iterations'
            % maxiters)
    return (False, maxiters, 0)




# ---- Test program --------

if __name__=='__main__':

    rs = RandomState(MT19937(SeedSequence(123456789)))

    level = 9
    A = mmread('TestMatrices/DH-Matrix-%d.mtx' % level)
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


    # Create a preconditioner
    drop = 1.0e-0
    print('Creating ILU preconditioner with drop tol = %g' % drop)
    precTimer = MyTimer('ILU creation')
    ILU = ILURightPreconditioner(A, drop_tol=drop, fill_factor=15)
    precTimer.stop()


    # Run GMRES
    print('Running GMRES')
    gmresTimer = MyTimer('GMRES')
    (conv, iters, x) = GMRES(A,b,maxiters=500, tol=1.0e-6, precond=ILU)
    gmresTimer.stop()

    # Delete the preconditioner object to save memory for the factorization
    del ILU

    # Print the error
    if conv:
        err = la.norm(x - xEx)/la.norm(xEx)
        print('\nGMRES relative error norm = %10.3g' % err)
    else:
        print('GMRES failed')


    # For comparison, do a sparse direct solve using SuperLU
    print('Running SuperLU')
    spluTimer = MyTimer('Super LU')
    LU = spla.splu(A.tocsc())
    xDirect = LU.solve(b)
    spluTimer.stop()

    err = la.norm(xDirect - xEx)/la.norm(xEx)
    print('\nSparse direct solve error norm = %10.3g' % err)

    print('\nTotal GMRES time (prec setup +iter)\t %10.3g seconds'
        % (precTimer.walltime()+gmresTimer.walltime()))
    print('\t-Preconditioner setup time:      %10.3g seconds'
        % precTimer.walltime())
    print('\t-GMRES iteration time:           %10.3g seconds'
        % gmresTimer.walltime())
    print('\nDirect solve time\t                %10.3g seconds'
        % spluTimer.walltime())
