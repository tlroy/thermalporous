from firedrake import *

class RectangleGeo():
    def __init__(self, Nx, Ny, params, Length = 365.76, Length_y = 365.76):
        self.Nx = Nx
        self.Ny = Ny
        self.dim = 2
        self.params = params
        self.Length = Length # default is 365.76m, i.e. 1200 ft for SPE10
        self.Length_y = Length_y
        self.mesh = self.generate_mesh(Nx, Ny)
        self.comm = self.mesh.comm
        self.init_function_space()
        self.generate_geo_fields() #defined in subclass
        
        try:
            self.K_x = self.K_x
            self.K_y = self.K_y
        except AttributeError:
            print("Isotropic permeability")
            self.K_x = self.K
            self.K_y = self.K 
        
    def generate_mesh(self, Nx, Ny):
        Length = self.Length
        Length_y = self.Length_y
        self.Dx = Length/Nx
        self.Dy = Length_y/Ny
        mesh = RectangleMesh(Nx, Ny, Length, Length_y, quadrilateral=True)
        return mesh
        
    def init_function_space(self):
        self.V = FunctionSpace(self.mesh, "DQ", 0)
