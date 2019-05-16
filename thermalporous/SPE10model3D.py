import numpy as np

from firedrake import *
from thermalporous.boxgeo import BoxGeo

class SPE10Model3D(BoxGeo):
    def __init__(self, Nx, Ny, Nz, params, save = False):
        self.geotype = "SPE10 " + str(Nx) + 'X' + str(Ny) + 'X' + str(Nz)
        self.name = self.geotype
        self.save = save
        Dx = 6.096
        Dy = 3.048
        Dz = 0.6096
        BoxGeo.__init__(self, Nx, Ny, Nz, params, Length = Nx*Dx, Length_y = Ny*Dy, Length_z = Nz*Dz) 
        
    def generate_geo_fields(self):
        import os
        dirname = os.path.dirname(__file__)

        Vvec = VectorFunctionSpace(self.mesh, "DQ", 0)
        coords = project(self.mesh.coordinates, Vvec).dat.data  
        
        # Creat DG function for porosity field
        self.phi = Function(self.V, name = 'porosity')
        # load data for porosity
        phi_array = np.load(dirname + "/../data/slice_phi.npy")
        
        phi_array = phi_array + 1e-10 #removing rock only cells
        
        def phi_coordtoijk(x, y, z):
            i = np.floor(x / self.Dx).astype(int)
            j = np.floor(y / self.Dy).astype(int)
            k = np.floor(z / self.Dz).astype(int)
            return phi_array[i, j, k]
        self.phi.dat.data[...] = phi_coordtoijk(coords[:, 0], coords[:, 1], coords[:, 2])
        
        self.K_x = Function(self.V, name = 'perm_x')
        # load data for permeability_x
        permx_array = np.load(dirname + "/../data/slice_perm_x.npy")
        
        def permx_coordtoijk(x, y, z):
            i = np.floor(x / self.Dx).astype(int)
            j = np.floor(y / self.Dy).astype(int)
            k = np.floor(z / self.Dz).astype(int)
            return permx_array[i, j, k]
        self.K_x.dat.data[...] = permx_coordtoijk(coords[:, 0], coords[:, 1], coords[:, 2])
        
        self.K_y = Function(self.V, name = 'perm_y')
        # load data for permeability_y
        permy_array = np.load(dirname + "/../data/slice_perm_y.npy")
        
        def permy_coordtoijk(x, y, z):
            i = np.floor(x / self.Dx).astype(int)
            j = np.floor(y / self.Dy).astype(int)
            k = np.floor(z / self.Dz).astype(int)
            return permy_array[i, j, k]
        self.K_y.dat.data[...] = permy_coordtoijk(coords[:, 0], coords[:, 1], coords[:, 2])
        
        self.K_z = Function(self.V, name = 'perm_z')
        # load data for permeability_z
        permz_array = np.load(dirname + "/../data/slice_perm_z.npy")
        
        def permz_coordtoijk(x, y, z):
            i = np.floor(x / self.Dx).astype(int)
            j = np.floor(y / self.Dy).astype(int)
            k = np.floor(z / self.Dz).astype(int)
            return permz_array[i, j, k]
        self.K_z.dat.data[...] = permz_coordtoijk(coords[:, 0], coords[:, 1], coords[:, 2])

        # Define conductivity field
        self.kT = Function(self.V, name = 'conductivity')
        self.kT = project(self.phi*self.params.ko + (1-self.phi)*self.params.kr, self.V)
        #print("Rock conductivity is: ", self.params.kr)
        
        if self.save is True:
            File("results/porosity.pvd").write(self.phi)
            File("results/permeability_x.pvd").write(self.K_x)
            File("results/permeability_y.pvd").write(self.K_y)
            File("results/permeability_z.pvd").write(self.K_z)
            File("results/conductivity.pvd").write(self.kT)
        
