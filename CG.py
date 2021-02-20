from numpy.linalg import norm
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import numpy as np
from scipy.io import mmread
from KrylovUtils import *

# Conjugate gradient iteration for solving a system Ax=b with A SPD
# * Matrix A (must be SPD)
# * RHS b
# * Maximum number of steps maxiter
# * Stopping tolerance tau (relative residual)
def CG(A, b, xEx, maxiter=100, tau=1.0e-8):

    # Get size of matrix
    n,nc = A.shape
    checkSquare(A, 'CG')

    # We'll need ||b|| for convergence testing
    normB = norm(b)

    # Check for the trivial case b=0, x=0
    if normB == 0.0:
        return (True, np.zeros_like(b))

    # Initialize the step, residual, and solution vectors
    p = 1.0*b # make a deep copy
    r = 1.0*b # make a deep copy
    x = np.zeros_like(b)

    # We'll need r^T * r.
    rDotR = np.dot(r,r) # Shouldn't be zero, since we've caught b=0

    # Main CG loop
    for k in range(maxiter):
        # Compute A*p
        Ap = mvmult(A,p)

        pTAp = np.dot(p,Ap)
        if pTAp==0.0:
            print('CG broke down: (p, Ap)=0')
            return (False, 0)

        alpha = rDotR/pTAp

        x = x + alpha*p
        r = r - alpha*Ap

        rDotRTmp = np.dot(r,r)
        normR = np.sqrt(rDotRTmp)
        normE = np.sqrt(np.dot((x-xEx), mvmult(A, (x-xEx))))
        print('\titer=%4d\t||r||=%12.5g ||e||_A=%12.5g'
            % (k, normR/normB, normE) )

        if normR <= tau*normB:
            print('CG Converged!')
            return (True, x)


        beta = rDotRTmp/rDotR
        rDotR = rDotRTmp

        p = r + beta*p

    # Failure to converge.
    return (False, x)


if __name__=='__main__':

    rs = RandomState(MT19937(SeedSequence(123456789)))

    level = 10
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


    print('\nRunning CG')
    maxiter = 1000
    tau=1.0e-8
    (conv, x) = CG(A, b, xEx, maxiter=maxiter, tau=tau)

    print('\nerror norm = %g' % (norm(x-xEx)/norm(xEx)))
