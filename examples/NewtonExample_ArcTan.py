from PySolvers.Linear import DefaultDirect
from PySolvers.Nonlinear import NewtonSolver, FuncAdapter1D
from PySolvers import CommonSolverArgs
import numpy as np

class ArcTanFunc(FuncAdapter1D):
    def __init__(self):
        pass

    def _evalF(self, x):
        return np.arctan(x)

    def _evalJ(self, x):
        return 1.0/(1.0 + x*x)



if __name__=='__main__':


    solver = NewtonSolver(control=CommonSolverArgs(tau=1.0e-15, maxiter=10),
                          solver=DefaultDirect(),
                          freezePrec=False)

    xInit = np.array([10,])

    stat = solver.solve(ArcTanFunc(), xInit)

    xEx = np.array([0,])

    if stat.success():
        x = stat.soln()
        err = solver.norm(x - xEx)
        print('err=%12.5g' % err)
    else:
        print('Solve failed: {}'.format(result.msg()))
