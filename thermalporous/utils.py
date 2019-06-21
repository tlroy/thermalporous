from firedrake import *
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

#def fprint(*output, filename = "results/results.txt"):
    #print(*output)
    #f = open(filename, "a")
    #print(*output, file = f)
    #with open("results/results.txt", "a") as f:
        #f.write("{}\n".format(*output))
