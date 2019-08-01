from firedrake import *
from petsc4py import *
import numpy as np

from mpi4py import MPI

def GetNodeClosestToCoordinate(V, coord):
    VV = VectorFunctionSpace(V.mesh(), V.ufl_element())
    coordinates = interpolate(SpatialCoordinate(V.mesh()), VV).vector()[:]
    # we need to do something clever in parallel if point is equidistant
    closest_dist = 100000
    closest_dof = 0
    
    for i in range(coordinates.shape[0]):
        dist = np.linalg.norm(coordinates[i,:]-coord)
        if dist < closest_dist:
            closest_dist = dist
            closest_dof = i
    comm = V.mesh().mpi_comm()
    closest_distance_all = comm.allreduce(closest_dist, MPI.MIN)
    if closest_dist == closest_distance_all:
        res = closest_dof
    else:
        res = -1
    return res

def ExportJacobian(F, u):
    J = derivative(F, u)
    JJ = assemble(J)#, mat_type = "aij")
    JJ.force_evaluation()
    A = JJ.petscmat

    myviewer = PETSc.Viewer().createASCII("matrix.txt") 
    myviewer.pushFormat(PETSc.Viewer.Format.ASCII_MATLAB) 
    A.view(myviewer)
    
def ExportResidual(F):
    b = assemble(F)

    with b.dat.vec_ro as w: 
        myviewer = PETSc.Viewer().createASCII("rhs.txt") 
        myviewer.pushFormat(PETSc.Viewer.Format.ASCII_MATLAB) 
        w.view(myviewer)

#def fprint(*output, filename = "results/results.txt"):
    #print(*output)
    #f = open(filename, "a")
    #print(*output, file = f)
    #with open("results/results.txt", "a") as f:
        #f.write("{}\n".format(*output))
