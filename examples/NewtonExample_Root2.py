from PySolvers.Linear import DefaultDirect
from PySolvers.Nonlinear import NewtonSolver, FuncAdapter1D
from PySolvers import CommonSolverArgs
import numpy as np

class Root2Func(FuncAdapter1D):
    def __init__(self):
        pass

    def _evalF(self, x):
        return x*x - 2

    def _evalJ(self, x):
        return 2.0*x



if __name__=='__main__':


    solver = NewtonSolver(control=CommonSolverArgs(tau=1.0e-15, maxiter=10),
                          solver=DefaultDirect())

    xInit = np.array([3,])

    stat = solver.solve(Root2Func(), xInit)

    xEx = np.array([np.sqrt(2),])

    if stat.success():
        x = stat.soln()
        err = solver.norm(x - xEx)
        print('err=%12.5g' % err)
    else:
        print('Solve failed: {}'.format(result.msg()))
