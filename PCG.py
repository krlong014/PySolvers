from numpy.linalg import norm
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import numpy as np
import scipy.sparse.linalg as spla
from scipy.io import mmread
from KrylovUtils import *
from BasicPreconditioner import PreconditionerBase, ILURightPreconditioner

# Preconditioned conjugate gradient iteration for solving a
# system Ax=b with A SPD
# * Matrix A (must be SPD)
# * RHS b
# * Maximum number of steps maxiter
# * Stopping tolerance tau (relative residual)
def PCG(A, b, maxiter=100, tau=1.0e-8, precond=PreconditionerBase()):

    # Get size of matrix
    n,nc = A.shape
    checkSquare(A, 'CG')

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


if __name__=='__main__':

    rs = RandomState(MT19937(SeedSequence(123456789)))

    level = 12
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



    maxiter = 30000
    tau=1.0e-14
    drop = 1.0e-4
    print('\nBuilding preconditioner')
    precTimer = MyTimer('Precond building')
    prec = ILURightPreconditioner(A, drop_tol=drop, fill_factor=30)
    prec = PreconditionerBase()
    precTimer.stop()

    print('\nRunning CG')
    cgTimer = MyTimer('CG loop')
    (conv, iters, x) = PCG(A, b, maxiter=maxiter, tau=tau, precond=prec)
    cgTimer.stop()


    print('\nerror norm = %g' % (norm(x-xEx)/norm(xEx)))

    # For comparison, do a sparse direct solve using SuperLU
    print('Running SuperLU')
    spluTimer = MyTimer('Super LU')
    LU = spla.splu(A.tocsc())
    xDirect = LU.solve(b)
    spluTimer.stop()

    err = norm(xDirect - xEx)/norm(xEx)
    print('\nSparse direct solve error norm = %10.3g' % err)

    print('\nTotal CG time (prec setup +iter)\t %10.3g seconds'
        % (precTimer.walltime()+cgTimer.walltime()))
    print('\t-Preconditioner setup time:      %10.3g seconds'
        % precTimer.walltime())
    print('\t-CG iteration time:           %10.3g seconds'
        % cgTimer.walltime())


    print('\nDirect solve time\t                %10.3g seconds'
        % spluTimer.walltime())
