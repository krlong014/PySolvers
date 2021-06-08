from numpy.linalg import norm
import numpy as np
import scipy.sparse.linalg as spla
from scipy.io import mmread
from . mvmult import mvmult
from . Preconditioner import Preconditioner, IdentityPreconditioner

# Preconditioned conjugate gradient iteration for solving a
# system Ax=b with A SPD
# * Matrix A (must be SPD)
# * RHS b
# * Maximum number of steps maxiter
# * Stopping tolerance tau (relative residual)
def PCG(A, b, maxiter=100, tau=1.0e-8, precond=IdentityPreconditioner()):

    # Get size of matrix
    n,nc = A.shape
    # Make sure it's square
    assert(n==nc)

    # Check for the trivial case b=0, x=0
    normB = norm(b)
    if normB == 0.0:
        return (True, np.zeros_like(b))

    # Initialize the step, residual, and solution vectors
    r = 1.0*b
    p = precond.applyRight(r)
    u = 1.0*p
    x = np.zeros_like(b)

    # We'll use ||b|| for computing relative residuals
    normB = norm(b)


    uDotR = np.dot(u,r) # Should never be zero, since we've caught b=0

    # Preconditioned CG
    for k in range(maxiter):
        # Compute A*p
        Ap = mvmult(A,p)

        pTAp = np.dot(p,Ap)
        if pTAp==0.0:
            print('PCG broke down: (p, Ap)=0')
            return (False, 0)

        # calculate step length
        alpha = uDotR/pTAp

        x = x + alpha*p
        r = r - alpha*Ap
        u = precond.applyRight(r)

        normR = norm(r)
        print('\titer=%4d\t||r||=%12.5g' % (k, normR/normB) )

        if normR <= tau*normB:
            print('PCG Converged!')
            return (True, k+1, x)

        newUDotR = np.dot(u,r)
        beta = newUDotR/uDotR
        uDotR = newUDotR

        p = u + beta*p

    # Failure to converge.
    return (False, maxiter, x)
