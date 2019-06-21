import numpy as np

from firedrake import *
from thermalporous.rectanglegeo import RectangleGeo

class SPE10Model(RectangleGeo):
    def __init__(self, Nx, Ny, params, save = False, plane = 'xy'):
        self.geotype = "SPE10"
        self.name = self.geotype
        self.save = save
        if plane == 'xy':
            Dx = 6.096
            Dy = 3.048
        elif plane == 'xz':
            print("xz slice")
            Dx = 6.096
            Dy = 0.6096
        elif plane == 'yz':
            Dx = 3.048
            Dy = 0.6096
        RectangleGeo.__init__(self, Nx, Ny, params, Length = Nx*Dx, Length_y = Ny*Dy) 
        
    def generate_geo_fields(self):
        import os
        dirname = os.path.dirname(__file__)

        Vvec = VectorFunctionSpace(self.mesh, "DQ", 0)
        coords = project(self.mesh.coordinates, Vvec).dat.data  
        
        # Create DG function for porosity field
        self.phi = Function(self.V, name = "porosity")
        phi_array = np.load(dirname + "/../data/slice_phi.npy")
        
        phi_array = phi_array + 1e-10 # removing rock only cells
        
        def phi_coordtoij(x, y):
            i = np.floor(x / self.Dx).astype(int)
            j = np.floor(y / self.Dy).astype(int)
            return phi_array[i, j]
        self.phi.dat.data[...] = phi_coordtoij(coords[:, 0], coords[:, 1])
        
        self.K_x = Function(self.V, name = "perm_x")
        # load data for permeability_x
        permx_array = np.load(dirname + "/../data/slice_perm_x.npy")
        
        def permx_coordtoij(x, y):
            i = np.floor(x / self.Dx).astype(int)
            j = np.floor(y / self.Dy).astype(int)
            return permx_array[i, j]
        self.K_x.dat.data[...] = permx_coordtoij(coords[:, 0], coords[:, 1])
        
        self.K_y = Function(self.V, name = "perm_y")
        # load data for permeability_y
        permy_array = np.load(dirname + "/../data/slice_perm_y.npy")
        
        def permy_coordtoij(x, y):
            i = np.floor(x / self.Dx).astype(int)
            j = np.floor(y / self.Dy).astype(int)
            return permy_array[i, j]
        self.K_y.dat.data[...] = permy_coordtoij(coords[:, 0], coords[:, 1])

        # Define conductivity field
        self.kT = Function(self.V)
        self.kT = project(self.phi*self.params.ko + (1-self.phi)*self.params.kr, self.V)
        
        if self.save is True:
            File("results/permeability_x.pvd").write(self.K_x)
            File("results/permeability_y.pvd").write(self.K_y)
            File("results/porosity.pvd").write(self.phi)
            File("results/conductivity.pvd").write(self.kT)
