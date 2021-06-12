import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import copy
from . MLHierarchy import MLHierarchy
from PyTab import Tab
from PyTimer import Timer

# Author: Nick Moore wrote the algorithms; Katharine Long wrote the object
# interface.
# Texas Tech University, 2021.

class SmoothedAggregationMLHierarchy(MLHierarchy):
    def __init__(self, A_fine, numLevels=2, tol=None, normalize=True):
        super().__init__(numLevels=numLevels, normalize=normalize)
        self.tol = tol
        self.normalize = normalize

        self._setFineMatrix(A_fine)
        for lev in reversed(range(numLevels-1)):
            I_up = self.makeProlongator(lev)
            self._setUpdate(lev, I_up)

    def makeProlongator(self, lev):
        '''Make the prologator to go from level k to k+1.'''
        tab = Tab()
        print('{}making prolongator from level {} to {}'.format(tab,lev,lev+1))
        (I_up, aggregates) = SA_coarsen(
                        self.matrix(lev+1), tol=self.tol, lvl=lev+1
                        )
        return I_up


# --------------------------------------------------------------------------
# Nick Moore's smoothed aggregation code





def getNeighborhood(A, i, tol, a_diag):
    tab = Tab()
    #print('{}in getNeighborhood()'.format(tab))
    N = {i}
    a_ii = a_diag[i]
    start = A.indptr[i]
    end = A.indptr[i+1]

    for k in range(start, end):
        j = A.indices[k]
        a_ij = A.data[k]
        a_jj = a_diag[j]
        if abs(a_ij) >= tol*np.sqrt(a_ii*a_jj):
            N.add(j)
    return N

def BuildAggregates(A, lvl=1, tol=None, phase=3):
    tab=Tab()
    tab1 = Tab()
    print('{}in BuildAggregates()'.format(tab))
    # If tol isn't specified, then use the default by Vanek
    if tol is None:
        tol = 0.08*(0.5)**(lvl-1)

    timer0 = Timer('BuildAggregates init step')
    timer0.start()
    # Initialization
    R = {i for i in range(A.shape[0])}
    a_diag = A.diagonal()
    neighborhoods = [getNeighborhood(A,i,tol,a_diag) for i in range(A.shape[0])]
    aggregates = []
    # isolated nodes aren't aggregated
    for n in neighborhoods:
        if len(n) == 1:
            aggregates.append(n)
            [elem] = n
            R.remove(elem)
    timer0.stop()

    timer1 = Timer('BuildAggregates phase 1')
    timer1.start()
    # Phase 1
    if phase > 0:
        for i in range(A.shape[0]):
            # If the neighborhood of i is completely in R, create an aggregate
            # from the neighborhood
            if i in R and neighborhoods[i].issubset(R):
                aggregates.append(neighborhoods[i])
                R -= neighborhoods[i]
    timer1.stop()

    timer2 = Timer('BuildAggregates phase 2')
    timer2.start()

    # Phase 2
    if phase > 1:
        # Copy the aggregates since we need to modify and check the
        # original aggregates
        timer_copy = Timer('agg copy')
        timer_copy.start()
        aggcopy = copy.deepcopy(aggregates)
        timer_copy.stop()
        # Loop through elements still in R
        for i in range(A.shape[0]):
            if i in R:
                # We need the strongest connection
                max_conn_strength = 0.0
                agg_idx_of_max = -1
                # Loop through all aggregates, looking to see if the current
                # neighborhood intersections
                for (j, agg) in enumerate(aggcopy):
                    #timer_disj = Timer('agg disjoint check')
                    #timer_disj.start()
                    isDisjoint_i_j = agg.isdisjoint(neighborhoods[i])
                    #timer_disj.stop()
                    if not isDisjoint_i_j:
                        #timer_loop = Timer('loop to find max strength')
                        #timer_loop.start()
                        for k in agg:
                            if abs(A[i,k]) > max_conn_strength:
                                max_conn_strength = abs(A[i,k])
                                agg_idx_of_max = j
                        #timer_loop.stop()
                timer_insert = Timer('agg insertion')
                timer_insert.start()
                aggregates[agg_idx_of_max].add(i)
                timer_insert.stop()

    timer2.stop()

    timer3 = Timer('BuildAggregates phase 2')
    timer3.start()

    # Phase 3
    if phase > 2 and not R:
        # Loop through elements still in R
        for i in range(A.shape[0]):
            if i in R:
                aggregates.append(R.intersection(neighborhoods[i]))
                R -= neighborhoods[i]
    timer3.stop()

    return (aggregates, neighborhoods)

def BuildTentativeProlongator(A, aggregates):
    tab=Tab()
    timer = Timer('BuildTentativeProlongator')
    timer.start()
    print('{}in BuildTentativeProlongator()'.format(tab))
    P = sp.dok_matrix((A.shape[0], len(aggregates)))
    for i in range(len(aggregates)):
        for j in aggregates[i]:
            P[j,i] = 1
    timer.stop()
    return P

def BuildFilteredMatrix(A, neighborhoods, tol):
    tab=Tab()
    print('{}in BuildFilteredMatrix()'.format(tab))
    timer = Timer('BuildFilteredMatrix')
    timer.start()
    # Expects A as csr
    Af = A.copy()
    for i in range(A.shape[0]):
        N = neighborhoods[i]

        start = Af.indptr[i]
        end = Af.indptr[i+1]

        for k in range(start, end):
            j = Af.indices[k]
            if i==j:
                iPtr = k
                break

        for k in range(start, end):
            j = Af.indices[k]
            if j not in N:
                Af.data[iPtr] -= Af.data[k]
                Af.data[k]=0

    timer.stop()
    return Af

def SmoothProlongator(Phat, A, Af, omega=(2/3)):
    tab=Tab()
    print('{}in SmoothProlongator()'.format(tab))
    timer = Timer('SmoothProlongator')
    timer.start()
    smoothmat = omega*Af
    d_A = A.diagonal()
    for i in range(A.shape[0]):
        start = smoothmat.indptr[i]
        end = smoothmat.indptr[i+1]
        for k in range(start, end):
            j = smoothmat.indices[k]
            smoothmat.data[k] /= d_A[i]
            if i==j:
                smoothmat.data[k] = 1 - smoothmat.data[k]
            else:
                smoothmat.data[k] = -smoothmat.data[k]

    smoothed = smoothmat.dot(Phat)
    timer.stop()
    return smoothed

# Build the SA prolongation operator for a matrix
def SA_coarsen(A, tol=None, lvl=1):
    tab=Tab()
    print('{}in SA_coarsen()'.format(tab))
    # lvl will only be used if tol is None

    # If tol is None, use Vanek's suggestion
    if tol is None:
        tol = 0.08*(0.5)**(lvl-1)

    # Build the aggregates
    (aggregates, neighborhoods) = BuildAggregates(A, lvl=lvl)

    # Build the tentative prolongator from the aggregates, A needed for it's dimensions
    Phat = BuildTentativeProlongator(A, aggregates)

    # Build the filtered matrix for the smoother
    Af = BuildFilteredMatrix(A, neighborhoods, tol)

    # Smooth the Prolongation Operator with weighted Jacobi using the filtered matrix
    P = SmoothProlongator(Phat, A, Af)

    return (P.tocsr(), aggregates)

if __name__=='__main__':

  from DebyeHuckel import DiscretizeDH, ConstantFunc
  from UniformRectangleMesher import UniformRectangleMesher
  from MPLMeshAggViewer import MPLMeshAggViewer

  mesh = UniformRectangleMesher(0.0, 1.0, 5, 0.0, 1.0, 4)
  #mesh = GoofySquare1()

  beta = 0.0
  load = ConstantFunc(beta*beta)
  (A,b) = DiscretizeDH(mesh, load, beta)

  colors = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4',
    '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990', '#dcbeff',
    '#9A6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075',
    '#a9a9a9', '#ffffff', '#000000']

  ( P, aggregates ) = SA_coarsen(A, tol=None, lvl=1)
  print(P)

  viewer = MPLMeshAggViewer(aggregates=aggregates, colors=colors, vertRad=0.05, fontSize=10)
  viewer.show(mesh)
