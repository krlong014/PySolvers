import scipy.sparse as sp
import numpy as np
from FDLaplacian2D import FDLaplacian2D

from PySolvers.Linear import PCG, RightIC, AMG
from PySolvers.Nonlinear import NewtonSolver
from PySolvers import CommonSolverArgs
from FDLaplacian2D import FDLaplacian2D

class FDBratu2D:
    def __init__(self, m=4, alpha=0.5):
        self.m = m
        self.alpha = alpha

        self.A = -FDLaplacian2D(-1.0, 1.0, m)

    def initialU(self):
        return np.ones(self.m*self.m)

    def evalF(self, u):
        return self.A*u - self.alpha*np.exp(-u)

    def evalJ(self, u):
        J = self.A.copy()
        g = self.alpha*np.exp(-u)
        d = J.diagonal()
        J.setdiag(d+g)

        return J



if __name__=='__main__':

    #prec = RightIC()
    prec = AMG(numIters=5)

    solver = NewtonSolver(control=CommonSolverArgs(tau=1.0e-12, maxiter=10),
                          solver=PCG(precond=prec),
                          fixLinTol=False,
                          minLinTol=1.0e-6,
                          freezePrec=True)


    func = FDBratu2D(m=100)
    xInit = func.initialU()

    stat = solver.solve(func, xInit)

    if stat.success():
        x = stat.soln()
        print('Success!')
    else:
        print('Solve failed: {}'.format(result.msg()))
