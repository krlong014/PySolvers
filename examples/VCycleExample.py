from PySolvers.Linear import AMGVCycle, CommonSolverArgs
from DHTestProblem import DHTestProblem
import numpy as np
import numpy.linalg as npla
import argparse

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Default direct solver example')
    parser.add_argument('--meshLev', type=int, action='store', default=10)
    parser.add_argument('--maxiter', type=int, action='store', default=100)
    parser.add_argument('--nupre', type=int, action='store', default=2)
    parser.add_argument('--nupost', type=int, action='store', default=2)
    parser.add_argument('--levels', type=int, action='store', default=2)
    parser.add_argument('--tau', type=np.double, action='store', default=1.0e-8)

    args = parser.parse_args()
    print('levels={}, type={}'.format(args.levels.__repr__(), type(args.levels)))

    (A,b,xEx) = DHTestProblem(args.meshLev)

    control = CommonSolverArgs(maxiter=args.maxiter, tau=args.tau)
    solverType = AMGVCycle(control=control, numLevels=args.levels,
                           nuPre=args.nupre, nuPost=args.nupost)
    solver = solverType.makeSolver()


    result = solver.solve(A, b)

    if result.success():
        x = result.soln()
        err = npla.norm(x - xEx)
        print('err=%12.5g' % err)
    else:
        print('Solve failed: {}'.format(result.msg()))
