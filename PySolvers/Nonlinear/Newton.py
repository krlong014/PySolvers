from .. IterativeSolver import (IterativeSolver, CommonSolverArgs)
from .. Linear.DefaultDirectSolver import DefaultDirect
from .. Linear.IterativeLinearSolver import IterativeLinearSolver
from . LineSearch import SimpleBacktrack
from . PreconditionerFreeze import PreconditionerFreeze
from PyTab import Tab
import numpy as np


class NewtonSolver(IterativeSolver):

    def __init__(self, control=CommonSolverArgs(),
                 solver = DefaultDirect(),  # Linear solver
                 linesearch = SimpleBacktrack(),
                 fixLinTol=False,
                 tolFudge=0.1,
                 minLinTol=1.0e-10,
                 freezePrec=True,
                 name='Newton'):
        super().__init__(control, name=name)
        self.solver = solver.makeSolver()
        self.linesearch = linesearch
        self.fixLinTol = fixLinTol
        self.tolFudge = tolFudge
        self.minLinTol = minLinTol
        self.freezePrec = freezePrec


    def solve(self, func, xInit):
        '''
        '''

        tab = Tab()
        xCur = xInit.copy()
        FCur = func.evalF(xCur)
        newtStep = np.ones_like(xCur)

        print('freeze prec for solver=', self.freezePrec)
        freeze = PreconditionerFreeze(self.solver, self.freezePrec)

        self.linesearch.setNorm(self.norm)

        # Initial residual; to be used for relative residual tests
        r0 = self.norm(FCur)
        normFCur = r0

        # Newton loop
        for i in range(self.maxiter()):

            # Report progress
            self.reportIter(i, normFCur, r0)

            # Check for convergence
            if normFCur <= r0*self.tau() + self.tau():
                return self.handleConvergence(i, xCur, normFCur, r0)


            # Evaluate Jacobian
            J = func.evalJ(xCur)

            # Set tolerance to be used in linear solve
            if isinstance(self.solver, IterativeLinearSolver):
                if self.fixLinTol: # Use fixed tolerance if desired (for testing)
                    tau_lin = self.minLinTol
                else: # Adjust linear tol according to nonlinear residual.
                    # new linear tolerance is the larger of:
                    # (*) "fudge factor" times relative residual of nonlinear solve
                    # (*) a minimum linear tolerance.
                    # The minimum tolerance avoids pointlessly small tolerances
                    # for the linear solver
                    tau_lin = max(self.tolFudge*normFCur/r0, self.minLinTol)

                self.solver.setTolerance(tau_lin)

            # Solve for the Newton step
            tab.indent()
            status = self.solver.solve(J, -FCur)
            tab.unindent()

            if not status.success():
                return self.handleBreakdown(i,
                    'solve for Newton step failed with msg={}'.
                    format(status.msg()))

            p = status.soln()

            # Do line search to find a step length giving sufficient decrease
            tab.indent()
            (success, xCur, FCur, normFCur) = self.linesearch.search(
                xCur, normFCur, p, func)
            tab.unindent()
            if not success:
                return self.handleBreakdown(i, msg='Line search failed')

        # End of Newton loop. At this point, xCur, FCur, and normFCur have been
        # updated to the new step.

        # If here, we've reached the maximum number of iterations without
        # convergence. Report failure to converge.

        return self.handleMaxiter(self.maxiter(), xCur, normFCur, r0)
