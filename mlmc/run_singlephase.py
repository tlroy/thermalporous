#!/usr/bin/env python

# MLMC tests using https://bitbucket.org/pefarrell/pymlmc/
# Make sure to add pymlmc to path
# Random sampling is done via Karhunen-Loeve (K-L) expansion
# For K-L, a generalized eigenvalue problem is solved. By default we use SLEPc.
# SLEPc can be installed using: python3 firedrake-install --slepc
# For large problems, 64-bit PETSc integers are needed for the eigenvalue problem.
# This is done by adding --petsc-int-type int64


from pymlmc import mlmc_test, mlmc_plot, mlmc_fn
import matplotlib.pyplot as plt
import numpy
from firedrake import *
from firedrake.petsc import PETSc
import petsc4py
import scipy
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import cKDTree

from thermalporous.physicalparameters import PhysicalParameters as Params
from thermalporous.homogeneousgeo import HomogeneousGeo as GeoModel
from thermalporous.wellcase import WellCase as TestCase
from thermalporous.singlephase import SinglePhase as ThermalModel

try:
    from slepc4py import SLEPc
except ImportError:
    import sys
    warning("Unable to import SLEPc, eigenvalue computation not possible (try firedrake-update --slepc)")
    sys.exit(0)
from time import time

def make_nested_mapping(outer_space, inner_space):
    # Maps the dofs between nested meshes
    outer_dof_coor = Function(VectorFunctionSpace(outer_space.mesh(), "DG", 0)).interpolate(SpatialCoordinate(outer_space.mesh())).vector()[:]
    inner_dof_coor = Function(VectorFunctionSpace(inner_space.mesh(), "DG", 0)).interpolate(SpatialCoordinate(inner_space.mesh())).vector()[:]

    tree = cKDTree(outer_dof_coor)
    _,mapping = tree.query(inner_dof_coor, k=1)
    return mapping

def covariance_func(X):
    # Gaussian covariance function
    # input X: array of size N_dofs X 2
    # output c: array of size N_dofs X N_dofs
    s = 0.25 # scaling parameter. correlation length
    pairwise_sq_dists = squareform(pdist(X, 'sqeuclidean'))
    c = numpy.exp(-pairwise_sq_dists / s**2)
    c = (c > 1e-16) * c
    return c

def karhunenloeve(problems, mappings, Nt):
    # Truncated Karhunen-Loeve expansion
    # We are solving the generalized eigenvalue problem M*C*M*phi= lmbda*M*phi
    # where M is the FEM mass matrix and C is the covariance matrix,
    # lmbda is an eigenvalue and phi the associated eigenmode.
    # Random field is approximated by sum_{i=1}^Nt sqrt(lmbda_i) phi_i 

    slepc = True

    V = problems[-1].V
    u = TrialFunction(V)
    v = TestFunction(V)
    a = u*v*dx
    petsc_m = assemble(a).M.handle

    n = V.dim()
    mesh = V.mesh()
    x = SpatialCoordinate(mesh)
    Vv = VectorFunctionSpace(mesh, "DG", 0)
    coords = interpolate(x, Vv).dat.data

    C = covariance_func(coords)

    phi = [Function(V) for i in range(Nt)]

    # Using scipy for the Generalized Eigenvalue problem
    if not slepc:
        mi, mj, mv = petsc_m.getValuesCSR()
        M = scipy.sparse.csr_matrix( (mv, mj, mi), shape = C.shape).todense()
        A = M.dot(C.dot(M))
        lmbda,v = scipy.linalg.eigh(A, M, eigvals = (V.dim()-Nt,V.dim()-1))
        for i in range(Nt):
            phi[i].vector()[:] = v[:,i]
    # Using SLEPc for the Generalized Eigenvalue problem
    if slepc:
        CC = scipy.sparse.csr_matrix(C)
        I = CC.indptr
        J = CC.indices
        data = CC.data
        del CC
        petsc_c = PETSc.Mat();
        # Use 'int64' if using int64 indices
        petsc_c.createAIJWithArrays([n,n], (I.astype('int32'), J.astype('int32'), data), comm = PETSc.COMM_WORLD);
        del I, J, data
        petsc_mc = petsc_m.matMult(petsc_c);
        del petsc_c
        petsc_a = petsc_mc.matMult(petsc_m);
        opts = PETSc.Options()
        opts.setValue("eps_gen_hermitian", None)
        opts.setValue("st_pc_factor_shift_type", "NONZERO")
        opts.setValue("eps_type", "krylovschur")
        opts.setValue("eps_largest_real", None)
        opts.setValue("eps_tol", 1e-10)

        es = SLEPc.EPS().create(comm=COMM_WORLD)
        es.setDimensions(Nt)
        es.setOperators(petsc_a, petsc_m)
        es.setFromOptions()
        es.solve()
        nconv = es.getConverged()
        lmbda = []
        vr, vi = petsc_a.getVecs()
        del petsc_a, petsc_m
        for i in range(Nt):
            lmbda.append(es.getEigenpair(i, vr, vi))
            phi[i].vector()[:] = vr

    lmbda = numpy.real(lmbda)
    print("Eigenvalue ratio: ", lmbda[0]/lmbda[-1])
    sqrtlmbda = numpy.sqrt(lmbda)
    
    phis = [[Function(problem.V) for i in range(Nt)] for problem in problems[0:-1]]
    for l in range(len(phis)):
        for i in range(Nt):
            phis[l][i].vector()[:] = phi[i].vector()[mappings[l]]
    phis.append(phi)
    H = [numpy.vstack([phis[l][i].vector()[:] for i in range(Nt)]) for l in range(len(problems))]

    def generate_field(l, gaussians):
        Kw = Function(problems[l].V)
        Kw.vector()[:] = numpy.exp((sqrtlmbda*gaussians).dot(H[l]))
        return Kw

    return generate_field

class DarcyProblem(object):
    # We are solving a single phase thermal oil problem with one production and one injection well.
    # The permeability field is log-normal.
    # Quantity of interest is production rate at time t=end

    # l: current level
    # M: refinement factor
    def __init__(self, l, M):

        # Setting up physical parameters
        params = Params()
        params.p_prod = params.p_prod*1e-1
        params.p_inj = params.p_inj*1e-1
        params.p_ref = params.p_ref*1e-1
        params.well_radius = 0.1

        # TIME STEPPING
        end = 0.5/24.
        num_steps = 1*M**l
        maxdt = end/num_steps
        # GEO
        N = int(numpy.ceil(5*M**l))
        geo = GeoModel(N, N, params, 1., 1.)
        self.Kw = Function(geo.V)
        geo.K_x = geo.K_x * self.Kw
        geo.K_y = geo.K_y * self.Kw
        geo.phi = geo.phi
        geo.kT = geo.phi*geo.params.ko + (1-geo.phi)*geo.params.kr


        # CASE
        prod_points = [[0.3,0.5]]
        inj_points = [[0.7,0.5]]
        case = TestCase(params, geo, prod_points = prod_points, inj_points = inj_points)
        case.wellfunc = 'square'
        case.init_wells(prod_points, inj_points, 'square')

        self.model = ThermalModel(geo, case, params, solver_parameters = "pc_fieldsplit_cd", end = end, maxdt = maxdt, small_dt_start = False, verbosity = False)

        self.V = self.model.V

    def _evaluate(self, sample):
        self.Kw.assign(sample)
        self.model.dt.assign(self.model.maxdt*24.*3600.)
        self.model.solve()
        well = self.model.case.prod_wells[0]
        P = 1e6*assemble(well['delta']*well['rate']*dx)
        return P

    def evaluate(self, sample):
        if isinstance(sample, list):
            n = len(sample)
            P = numpy.zeros(n)
            for i in range(n):
                P[i] = self._evaluate(sample[i])
            return P
        else:
            return self._evaluate(sample)
        
if __name__ == "__main__":
    N0 = 10 # initial samples on coarse levels
    Lmin = 2  # minimum refinement level
    Lmax = 5 # maximum refinement level
    M = 2 # refinement factor
    N = 100 # samples for convergence tests
    L = 5 # levels for convergence tests
    Nt = 80 # Karhunen-Loeve truncation
    Eps = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005]
    filename = "singlephase.txt"
    logfile = open(filename, "w")

    name = "Thermal single-phase reservoir simulation  with random field diffusion coefficent"
    print('\n ---- ' + name + ' ---- \n')
    l_range = range(Lmax+1)
    print('\n***** Generating problems *****\n')
    problems = [DarcyProblem(l, M) for l in l_range]
    print('\n***** Generating K-L expansion *****\n')
    twolevel_mappings = [None] + [make_nested_mapping(problems[l+1].V, problems[l].V) for l in l_range[0:-1]]
    top_mappings = [make_nested_mapping(problems[-1].V, problems[l].V) for l in l_range[0:-1]]
    generate_field = karhunenloeve(problems, top_mappings, Nt)
    sig = 1.0 # standard deviation
    def sampler(N, l):
        samplesf = []
        samplesc = []
        for i in range(N):
            gaussians = numpy.sqrt(sig)*numpy.random.randn(Nt)
            samplef = generate_field(l, gaussians)
            samplesf.append(samplef)
            if l > 0:
                samplec = Function(problems[l-1].V)
                samplec.vector()[:] = samplef.vector()[twolevel_mappings[l]] # map field to coarse level
                samplesc.append(samplec)
        return samplesf, samplesc
    def darcyfield_l(l, N):
        return mlmc_fn(l, N, problems, sampler = sampler)

    mlmc_test(darcyfield_l, N, L, N0, Eps, Lmin, Lmax, logfile)
    del logfile
    mlmc_plot(filename, nvert=3)
    plt.savefig(filename.replace('.txt', '.eps'))
