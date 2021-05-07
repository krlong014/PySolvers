import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import copy

def getNeighborhood(A, i, tol):
    N = {i}

    for j in range(A.shape[1]):
        if abs(A[i,j]) >= tol*np.sqrt(A[i,i]*A[j,j]):
            N.add(j)
    return N

def BuildAggregates(A, lvl=1, tol=None, phase=3):
    # If tol isn't specified, then use the default by Vanek
    if tol is None:
        tol = 0.08*(0.5)**(lvl-1)

    # Initialization
    R = {i for i in range(A.shape[0])}
    neighborhoods = [getNeighborhood(A,i,tol) for i in range(A.shape[0])]
    aggregates = []
    # isolated nodes aren't aggregated
    for n in neighborhoods:
        if len(n) == 1:
            aggregates.append(n)
            [elem] = n
            R.remove(elem)

    # Phase 1
    if phase > 0:
        for i in range(A.shape[0]):
            # If the neighborhood of i is completely in R, create an aggregate from the neighborhood
            if i in R and neighborhoods[i].issubset(R):
                aggregates.append(neighborhoods[i])
                R -= neighborhoods[i]

    # Phase 2
    if phase > 1:
        # Copy the aggregates since we need to modify and check the original aggregates
        aggcopy = copy.deepcopy(aggregates)
        # Loop through elements still in R
        for i in range(A.shape[0]):
            if i in R:
                # We need the strongest connection
                max_conn_strength = 0.0
                agg_idx_of_max = -1
                # Loop through all aggregates, looking to see if the current neighborhood intersections
                for (j, agg) in enumerate(aggcopy):
                    if not agg.isdisjoint(neighborhoods[i]):
                        for k in agg:
                            if abs(A[i,k]) > max_conn_strength:
                                max_conn_strength = abs(A[i,k])
                                agg_idx_of_max = j
                aggregates[agg_idx_of_max].add(i)

    # Phase 3
    if phase > 2 and not R:
        # Loop through elements still in R
        for i in range(A.shape[0]):
            if i in R:
                aggreagates.append(R.intersection(neighborhoods[i]))
                R -= neighborhoods[i]

    return aggregates

def BuildTentativeProlongator(A, aggregates):
    P = np.zeros((A.shape[0], len(aggregates)))
    for i in range(len(aggregates)):
        for j in aggregates[i]:
            P[j,i] = 1
    return P

def BuildFilteredMatrix(A, tol):
    # Expects A as a lil so I can subscript it
    # Could be improved to take a different format, but I was lazy
    Af = A.copy()
    for i in range(A.shape[0]):
        N = getNeighborhood(A, i, tol)
        for j in range(A.shape[1]):
            if j not in N:
                Af[i,i] -= Af[i,j]
                Af[i,j] = 0
    return Af

def SmoothProlongator(Phat, A, Af, omega=(2/3)):
    smoothmat = omega*Af
    d_A = A.diagonal()
    for i in range(A.shape[0]):
        smoothmat[i,:] /= d_A[i]

    smoothmat = np.eye(A.shape[0]) - smoothmat
    return smoothmat.dot(Phat)

# Build the SA prolongation operator for a matrix 
def SA_coarsen(A, tol=None, lvl=1):
    # lvl will only be used if tol is None

    # If tol is None, use Vanek's suggestion
    if tol is None:
        tol = 0.08*(0.5)**(lvl-1)

    # Build the aggregates
    aggregates = BuildAggregates(A, lvl=lvl)

    # Build the tentative prolongator from the aggregates, A needed for it's dimensions
    Phat = BuildTentativeProlongator(A, aggregates)
    
    # Build the filtered matrix for the smoother (expects lil matrix because I was lazy)
    Af = BuildFilteredMatrix(A.tolil(), tol)

    # Smooth the Prolongation Operator with weighted Jacobi using the filtered matrix
    P = SmoothProlongator(Phat, A, Af)

    return (P, aggregates)

if __name__=='__main__':

  from DebyeHuckel import DiscretizeDH, ConstantFunc
  from UniformRectangleMesher import UniformRectangleMesher
  from MPLMeshAggViewer import MPLMeshAggViewer

  mesh = UniformRectangleMesher(0.0, 1.0, 5, 0.0, 1.0, 4)
  #mesh = GoofySquare1()

  beta = 0.0
  load = ConstantFunc(beta*beta)
  (A,b) = DiscretizeDH(mesh, load, beta)

  colors = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9', '#ffffff', '#000000']

  ( P, aggregates ) = SA_coarsen(A, tol=None, lvl=1) 
  print(P)

  viewer = MPLMeshAggViewer(aggregates=aggregates, colors=colors, vertRad=0.05, fontSize=10)
  viewer.show(mesh)

