import pytest
from PySolvers import PCG
from PySolvers import ICRightPreconditioner
from PySolvers import mvmult
from scipy.io import mmread
from numpy.linalg import norm
import numpy.random

class TestPCG:

    def test_PCG(self):

        rng = numpy.random.default_rng()

        # Read a matrix
        level = 10
        A = mmread('../TestMatrices/DH-Matrix-%d.mtx' % level)
        A = A.tocsr()
        n,nc = A.shape
        print('Matrix is {} by {}'.format(n,nc))


        # Create a solution vector
        xEx = rng.random(n)
        # Multiply the solution by A to produce a RHS vector
        b = mvmult(A, xEx)

        maxiter = 100
        tau_r = 1.0e-10
        tau_a = 1.0e-8

        prec = ICRightPreconditioner(A)

        (conv, iters, x) = PCG(A, b, maxiter=maxiter, tau=tau_r, precond=prec)

        # Compute the norm of the error
        e_a = norm(x - xEx)

        normB = norm(b)
        assert(e_a <= tau_a)
