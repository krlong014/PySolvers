from PySolvers.Linear import PCG, RightIC, CommonSolverArgs
from DHTestProblem import DHTestProblem
import numpy as np
import numpy.linalg as npla
import argparse

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Default direct solver example')
    parser.add_argument('--meshLev', type=int, action='store', default=10)
    parser.add_argument('--maxiter', type=int, action='store', default=100)
    parser.add_argument('--tau', type=np.double, action='store', default=1.0e-8)

    args = parser.parse_args()

    (A,b,xEx) = DHTestProblem(args.meshLev)

    control = CommonSolverArgs(maxiter=args.maxiter, tau=args.tau,
                               precond=RightIC())
    solverType = PCG(args=control)
    solver = solverType.makeSolver()


    result = solver.solve(A, b)

    if result.success():
        x = result.soln()
        err = npla.norm(x - xEx)
        print('err=%12.5g' % err)
    else:
        print('Solve failed: {}'.format(result.msg()))
