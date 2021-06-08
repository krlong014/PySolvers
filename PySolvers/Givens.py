import numpy.linalg as la
import numpy as np


# Find the Givens coefficients c, s such that the rotation
# [[c, s], [-s, c]] will zero out the i+1 element of x.
def findGivensCoefficients(x, i):
    hyp = np.sqrt(x[i+1]*x[i+1] + x[i]*x[i])
    s = x[i+1]/hyp
    c = x[i]/hyp

    return (c, s)


# Apply a Givens rotation at the rows [i:i+2] of the vector x
def applyGivens(x, c, s, i):

    xi = x[i]
    xi1 = x[i+1]

    x[i] = c*xi + s*xi1
    x[i+1] = -s*xi + c*xi1

    return x

# Apply a Givens rotation at the rows [i:i+2] of the vector x. The
# input vector is modified.
def applyGivensInPlace(x, c, s, i):

    xi = x[i]
    xi1 = x[i+1]

    x[i] = c*xi + s*xi1
    x[i+1] = -s*xi + c*xi1

if __name__=='__main__':

    print('======= Test of Givens reduction from Hessenberg to triangular')
    # Test Hessenberg matrix
    H = np.array([
        [1.0, -2.0, 3.0],
        [4.0, 5.0, 6.0],
        [0.0, 7.0, 8.0],
        [0.0, 0.0, 9.0]
    ])

    n,m = H.shape # Here we use Saad's convention: n=rows, m=cols

    print('\n %d by %d Hessenberg matrix H= \n' %(n,m), H)
    print('\n||H||=', la.norm(H))
    initialNormH = la.norm(H)


    # Test RHS
    g = np.array([1.0,0.0,0.0,0.0])

    print('\ng= \n', g)
    initialNormRHS = la.norm(g)
    print('\n||g||=', initialNormRHS)

    # For comparison, compute a QR factorization using Numpy's built-in
    # function. This uses a sequence of Householder reflections.
    QR = la.qr(H, 'reduced')
    print('\n\n--- Numpy\'s QR factorization of H\n')
    print('Q=\n', QR[0])
    print('\nR=\n', QR[1])

    # For comparison, use normal equations to find the least squares solution
    # to H*y = g.
    yLS = la.solve(np.matmul(H.transpose(), H), np.matmul(H.transpose(), g))

    print('\n\n--- Solution of least squares problem via normal equations\n')
    print(yLS)

    print('\nResidual\n')
    r = g - np.matmul(H,yLS)
    residNorm = la.norm(r)
    print(residNorm)


    # Do Givens rotations to transform H to upper triangular form. At the
    # same time, apply the same operations to the RHS
    print('\n\n--- Doing Givens rotations on H')
    # Create an array in which we'll store all the rotation cosines and sines
    CS = np.zeros([m,2])

    # Loop over columns of H
    for i in range(0,m):
        # Apply the previous Givens rotations to the current column
        for j in range(0,i):
            #H[:,i] = applyGivens(H[:,i], CS[j,0], CS[j,1], j)
            applyGivensInPlace(H[:,i], CS[j,0], CS[j,1], j)
        # Find the Givens rotation that will zap the single subdiagonal entry
        # in the current column
        CS[i,:] = findGivensCoefficients(H[:,i], i)
        # Apply that rotation to zap the subdiagonal
        #H[:,i] = applyGivens(H[:,i], CS[i,0], CS[i,1], i)
        applyGivensInPlace(H[:,i], CS[i,0], CS[i,1], i)
        # Apply the same rotation to the RHS of the least squares problem
        #g = applyGivens(g, CS[i,0], CS[i,1], i)
        applyGivensInPlace(g, CS[i,0], CS[i,1], i)


    print('\nH after Givens procedure=\n')
    print(H)
    print('\n||H||=', la.norm(H))
    print('\nRHS after Givens procedure')
    print('g=\n', g)
    print('\n||g||=', la.norm(g))

    print('\nRelative error in norm:')
    print('H error: ', np.abs(la.norm(H)-initialNormH)/initialNormH)
    print('RHS error: ', np.abs(la.norm(g)-initialNormRHS)/initialNormRHS)

    # Solve the triangular system
    print('\n\n--- Triangular solve')
    y = la.solve(H[0:m,:], g[0:m])
    print('\nSolution: y=\n', y)

    print('For comparison, y from normal equations=\n', yLS)
