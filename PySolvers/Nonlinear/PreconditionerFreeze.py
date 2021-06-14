from .. Linear.IterativeLinearSolver import IterativeLinearSolver

class PreconditionerFreeze:
    '''
    This is a class that manages freezing and unfreezing of preconditioners
    during nonlinear solves. Unfreezing through the __del__() method of this
    class ensures that the preconditioner is unfrozen regardless of how
    the solver function is departed. 
    '''
    def __init__(self, solver, freezePrec):
        self.solver = solver
        self.freezePrec = freezePrec
        self.freeze()

    def freeze(self):
        if self.freezePrec and isinstance(self.solver, IterativeLinearSolver):
            self.solver.freezePrec()

    def unfreeze(self):
        if self.freezePrec and isinstance(self.solver, IterativeLinearSolver):
            self.solver.unfreezePrec()

    def __def__(self):
        self.unfreeze()
