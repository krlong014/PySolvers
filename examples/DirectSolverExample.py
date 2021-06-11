from PySolvers.Linear import DefaultDirect
from DHTestProblem import DHTestProblem
import numpy.linalg as npla
import argparse

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Default direct solver example')
    parser.add_argument('--meshLev', type=int, action='store', default=10)

    args = parser.parse_args()

    (A,b,xEx) = DHTestProblem(args.meshLev)

    solverType = DefaultDirect()
    solver = solverType.makeSolver()


    result = solver.solve(A, b)

    if result.success():
        x = result.soln()
        err = npla.norm(x - xEx)
        print('err=%12.5g' % err)
    else:
        print('Solve failed: {}'.format(result.msg()))
