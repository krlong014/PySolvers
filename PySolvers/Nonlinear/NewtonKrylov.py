import numpy.linalg as npla
import scipy.sparse.linalg as spla
import numpy as np
import scipy.sparse as sp
from Tab import Tab
from BasicPreconditioner import ILURightPreconditioner
from GMRES import GMRES
from FDBratu2D import FDBratu2D


class NewtonKrylov:
    def __init__(self,
                maxIters=20,            # maximum Newton iterations
                tau_r=1.0e-14,          # relative residual tolerance for Newton
                tau_a=1.0e-14,          # absolute residual tolerance for Newton
                maxLinIters=500,        # maximum iterations in linear solve
                minLinTol=1.0e-8,       # minimum linear solve tolerance
                tolFudge=0.1,           # multiplier for tol adjustment
                fixLinTol=False,        # whether to override tol adjustment
                reusePrecond=False,      # whether to reuse initial precond
                iluDrop=1.0e-4,         # drop tolerance for ILU
                iluFill=15,             # fill allowance for ILU
                verb=1,                 # verbosity for nonlinear solve
                linVerb=1):             # verbosity for linear solve


        self.maxIters = maxIters
        self.tau_r = tau_r
        self.tau_a = tau_a
        self.maxLinIters = maxLinIters
        self.minLinTol = minLinTol
        self.tolFudge = tolFudge
        self.fixLinTol = fixLinTol
        self.reusePrecond = reusePrecond
        self.iluDrop = iluDrop
        self.iluFill = iluFill
        self.verb = verb
        self.linVerb = linVerb

    def describe(self):
        tab0 = Tab()
        tab1 = Tab()
        print(tab0, '='*60)
        print(tab0, 'Newton-Krylov solver')
        print(tab1, 'Max iters: ', self.maxIters)
        print(tab1, 'Tolerance: tau_r=%12.5g, tau_a=%12.5g' %
            (self.tau_r, self.tau_a))
        print(tab1, 'Verbosity: ', self.verb)
        print(tab1, 'Newton-Krylov tolerance multiplier=%12.5g, min tol=%12.5g'%
            (self.tolFudge, self.minLinTol))
        print(tab1, 'use tolerance adjustment: ', not self.fixLinTol)
        print(tab1, 'Linear solver: GMRES, maxIters=', self.maxLinIters)
        print(tab1, 'Preconditioner: ILU(drop=%12.5g, fill=%d)' %
            (self.iluDrop, self.iluFill))
        print(tab1, 'Recycle initial preconditioner ', self.reusePrecond)


    def solve(self, func, uInit):

        # Report all parameters
        if self.verb>0:
            self.describe()

        # Set up formatting
        tab0 = Tab()
        tab1 = Tab()

        # Make a copy of the initial estimate
        u0 = uInit.copy()

        # Evaluate residual and its norm at initial iterate
        F0 = func.evalF(u0)
        r0 = npla.norm(F0)

        # Newton step vector. Initialize to all ones (this will be overwritten
        # before use)
        du = np.ones_like(u0)

        # We'll keep a count of the total Krylov iterations
        totalKrylovIters = 0

        # Run the loop!
        if self.verb>0:
            print('\n', tab0, 'Newton-Krylov loop')

        for i in range(self.maxIters):

            # Evaluate residual at current iterate (already done if i=0)
            if i>0:
                F0 = func.evalF(u0)
            # Compute residual norm
            r = npla.norm(F0)

            # Output convergence information if desired
            if self.verb>0:
                print(tab1, 'iter %6d r=%12.5g r/r0=%12.5g dx=%12.5g'
                    % (i, r, r/r0, npla.norm(du)))

            # Check for convergence
            if r <= r0*self.tau_r + self.tau_a:
                print(tab0, 'Converged!')
                print(tab0, 'totalKrylovIters=', totalKrylovIters)
                return (True, u0)

            # Compute Jacobian at current iterate
            J = func.evalJ(u0)

            # Update tolerance for linear solve
            if self.fixLinTol: # Use fixed tolerance if desired (for testing)
                tau_lin = self.minLinTol
            else: # Adjust linear tol according to nonlinear resid
                # new linear tolerance is the larger of:
                # (*) "fudge factor" times relative residual of nonlinear solve
                # (*) a minimum linear tolerance.
                # The minimum tolerance avoids pointlessly small tolerances
                # for the linear solver
                tau_lin = max(self.tolFudge*r/r0, self.minLinTol)

            # Update the preconditioner if desired
            if i==0 or not self.reusePrecond:
                print(tab1, 'Building ILU prec')
                drop = self.iluDrop
                fill = self.iluFill
                ILU = ILURightPreconditioner(J, drop_tol=drop, fill_factor=fill)


            (conv,krylovIters,du)=GMRES(J, -F0,
                maxiters=self.maxLinIters, tol=tau_lin, verb=self.linVerb,
                precond=ILU)
            totalKrylovIters += krylovIters

            if not conv:
                if self.verb>0:
                    print('Newton-Krylov: linear solver failed to converge')
                return (False, u0)

            # Update solution estimate
            u0 = u0 + du

        if self.verb>0:
            print(tab0, 'Newton-Krylov failed to converge!')
        return (False, u0)


if __name__=='__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Newton-Krylov')
    parser.add_argument('--fixtol', action='store', default=False)
    parser.add_argument('--reusePrec', action='store', default=True)
    parser.add_argument('--m', action='store', default=128)
    parser.add_argument('--tau_r', action='store', default=1.0e-14)
    parser.add_argument('--tau_a', action='store', default=1.0e-14)
    parser.add_argument('--fudge', action='store', default=0.05)
    parser.add_argument('--ilu_drop', action='store', default=1.0e-4)
    parser.add_argument('--tau_min', action='store', default=1.0e-8)

    args = parser.parse_args()

    solver = NewtonKrylov(fixLinTol=args.fixtol,
        tau_r=np.double(args.tau_r),
        tau_a=np.double(args.tau_a),
        tolFudge=np.double(args.fudge),
        reusePrecond=False,
        minLinTol=np.double(args.tau_min),
        iluDrop=np.double(args.ilu_drop))

    m = int(args.m)
    print('grid is %d by %d' % (m,m))
    func = FDBratu2D(m=m)
    u = func.initialU()

    (conv, uSoln) = solver.solve(func, u)
