from numpy.linalg import norm
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import numpy as np

# Arnoldi orthogonalization with Gram-Schmidt
# Input:
# * Matrix A
# * Initial vector v0
# * Number of steps m
def ArnoldiGS(A, v0, m):

    # Get size of matrix
    n,nc = A.shape
    # Add a check to ensure that n==nr

    # Pre-allocate Q and HBar
    Q = np.zeros([n, m+1])
    HBar = np.zeros([m+1, m])

    # Set first column of Q to the normalized initial vector
    Q[:,0] = v0 / norm(v0)

    # Arnoldi/GS loop
    for k in range(m):
        # Compute A q_k and store the result
        u = np.dot(A,Q[:,k])
        # Make a deep copy of u (so that we have a modifiable copy)
        v = np.copy(u)
        # Do inner GS loop
        for j in range(k+1):
            # Compute projection of u onto q_j
            HBar[j,k]=np.dot(Q[:,j], u)
            # Subtract off that projection (making changes to copy)
            v -= HBar[j,k]*Q[:,j]

        # Done with GS loop, compute h_{k+1,k}
        HBar[k+1,k]=norm(v)

        # Check for breakdown: happens when k = deg(min poly(A))
        if abs(HBar[k+1,k]) <= 1.0e-16:
            # If breakdown occurs, we're done
            print('Breakdown of Arnoldi at k=%d', k)
            return (Q, HBar)
        else:
            # Append new column to Q, and continue with algorithm
            Q[:,k+1]=v/HBar[k+1,k]

    # Done with m steps. Return results
    return (Q,HBar)

def ArnoldiMGS(A, v0, m):

    # Get shape
    n,nc = A.shape

    # Allocate space for results
    Q = np.zeros([n, m+1])
    HBar = np.zeros([m+1, m])

    # Initialize q_0
    Q[:,0] = v0 / norm(v0)

    # Outer Arnoldi loop
    for k in range(m):
        # Form A*q_k
        u = np.dot(A,Q[:,k])

        # Inner MGS loop
        for j in range(k+1):
            HBar[j,k]=np.dot(Q[:,j], u)
            u -= HBar[j,k]*Q[:,j]

        # Same as in GS
        HBar[k+1,k]=norm(u)

        if abs(HBar[k+1,k]) <= 1.0e-16:
            print('Breakdown of Arnoldi at k=%d', k)
            return (Q, HBar)
        else:
            Q[:,k+1]=u/HBar[k+1,k]

    return (Q,HBar)



def chop(a, eps=1.0e-14):

    with np.nditer(a, op_flags=['readwrite']) as it:
        for x in it:
            if abs(x) < eps:
                y = 0
            else:
                y = x
            x[...] = y
        return a

if __name__=='__main__':

    rs = RandomState(MT19937(SeedSequence(123456789)))

    n = 6
    A = rs.rand(n,n)
    normA = norm(A)

    if n <= 20: print('A=\n', A)

    v0 = rs.rand(n,1)[:,0]
    if n <= 20: print('v0=\n', v0)

    m = 4
    QPlus,HBar = ArnoldiGS(A, v0, m)

    if m <= 20: print('Q=\n', chop(QPlus))
    if m <= 20: print('HBar=\n', chop(HBar))

    Q = QPlus[:,0:m]
    if m <= 20: print('Q^T Q=\n', chop(np.dot(np.transpose(QPlus), QPlus)))
    H = HBar[0:m,:]
    if m <= 20: print('H=\n', H)
    AQ=np.dot(A,Q)

    QHBar=np.dot(QPlus, HBar)

    if n <= 20: print('A*Q=\n', chop(AQ))
    if n <= 20: print('Q*HBar=\n', chop(QHBar))

    delta = AQ-QHBar
    orthError = norm(np.dot(np.transpose(QPlus), QPlus) - np.eye(m+1))

    if n <= 20: print('A*Q-Q*H=\n', chop(delta))
    print('factorization error norm = %g' % (norm(delta)/normA))
    print('orthogonality error norm = %g' % norm(orthError))
