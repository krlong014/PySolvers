from PySolver.AMGSolver import AMGSolver
import DHTestProblem
import numpy.linalg as npla
import argparse

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='AMG solver parameters')
    parser.add_argument('--nuPrev', type=int, action='store', default=2)
    parser.add_argument('--nuPost', type=int, action='store', default=2)
    parser.add_argument('--levels', type=int, action='store', default=2)
    parser.add_argument('--maxiters', type=int, action='store', default=100)
    parser.add_argument('--tol', type=int, action='store', default=1.0e-8)
    parser.add_argument('--meshLev', type=int, action='store', default=10)

    args = parser.parse_args()

    (A,b,xEx) = DHTestProblem(meshLev)

    solver = AMGSolver(
        numLevels=args.levels,
        nuPrev=args.nuPrev,
        nuPost=args.nuPost,
        maxiters=args.maxiters,
        tol=args.tol)

    x = solver.solve(A, b)

    err = npla.norm(x - xEx)

    print('err=%12.5g' % err)
