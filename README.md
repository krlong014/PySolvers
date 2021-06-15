# PySolvers

A package with linear solvers, preconditioners, and nonlinear solvers. These
were developed during my course sequence on iterative solvers offered
at Texas Tech in 2020-21.

* Linear solvers
    * Preconditioned GMRES
    * Preconditioned conjugate gradients (PCG)
    * Algebraic multigrid (AMG)
        * The initial version of the AMG code was written by Nick Moore as part
        of his course project. 
    * Unified user interface to iterative and direct solvers
* Preconditioners
    * Incomplete ILU with drop tolerance (based on the numpy interface to SuperLU)
    * Incomplete Cholesky with drop tolerance (based on the numpy interface to SuperLU)
    * Algebraic multigrid (AMG)
* Nonlinear solvers
    * Newton's method
        * Inexact solves of linear subproblems with adaptive tolerance
        * Optional reuse of preconditioner for multiple linear subproblems
        * Line search to find sufficient decrease
