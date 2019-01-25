from firedrake import *

class BoxGeo():
    def __init__(self, Nx, Ny, Nz, params, Length = 365.76, Length_y = 365.76, Length_z = 1.8288):
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.dim = 3
        self.params = params
        self.Length = Length # default is 365.76m, i.e. 1200 ft for SPE10
        self.Length_y = Length_y
        self.Length_z = Length_z
        #self.Length = 10.0
        self.mesh = self.generate_mesh(Nx, Ny, Nz)
        self.comm = self.mesh.comm
        self.init_function_space()
        self.generate_geo_fields() #defined in subclass
        
        try:
            self.K_x = self.K_x
            self.K_y = self.K_y
            self.K_z = self.K_z
        except AttributeError:
            print("Isotropic permeability")
            self.K_x = self.K
            self.K_y = self.K 
            self.K_z = self.K 
        
    def generate_mesh(self, Nx, Ny, Nz):
        Length = self.Length
        Length_y = self.Length_y
        Length_z = self.Length_z
        self.Dx = Length/Nx
        self.Dy = Length_y/Ny
        self.Dz = Length_z/Nz
        # SPE 10 
        #Dx = 6.096
        #Dy = 3.048
        #Dz = 0.6096
        meshbase = RectangleMesh(Nx, Ny, Length, Length_y, quadrilateral=True)
        mesh = ExtrudedMesh(meshbase, Nz, self.Dz)
        return mesh
        
    def init_function_space(self):
        self.V = FunctionSpace(self.mesh, "DQ", 0)
        #self.VT = FunctionSpace(self.mesh, "DQ", 0)
        #self.W = self.V*self.VT   
