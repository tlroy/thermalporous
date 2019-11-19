from firedrake import *
from firedrake.petsc import *

class RectangleGeo():
    def __init__(self, Nx, Ny, params, Length = 365.76, Length_y = 365.76, mg = {}):
        self.Nx = Nx
        self.Ny = Ny
        self.dim = 2
        self.params = params
        self.Length = Length # default is 365.76m, i.e. 1200 ft for SPE10
        self.Length_y = Length_y
        if bool(mg):
            self.mesh = self.generate_mesh_hierarchy(Nx, Ny, mg["nref"])
        else:
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
    
    def generate_mesh_hierarchy(self, Nx, Ny, nref):
        Length = self.Length
        Length_y = self.Length_y
        if (Nx % 2**nref == 0) is False:
            print("WARNING: Nx is not divisible by nref. Rounding up")
            nx = int(Nx/2**nref) + 1
            Nx = nx*2**nref
            self.Nx = Nx
        else:
            nx = Nx/2**nref
        if (Ny % 2**nref == 0) is False:
            print("WARNING: Ny is not divisible by nref. Rounding up")
            ny = int(Ny/2**nref) + 1
            Ny = ny*2**nref
            self.Ny = Ny
        else:
            ny = Ny/2**nref

        self.Dx = Length/Nx
        self.Dy = Length_y/Ny
            
        distribution_parameters={"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)} # or FACET?
        #from IPython import embed; embed()
        base = RectangleMesh(nx, ny, Length, Length_y, quadrilateral=True, distribution_parameters=distribution_parameters)
        mh = MeshHierarchy(base, nref)
        mesh = mh[-1]
        return mesh
        
    
        
    def init_function_space(self):
        self.V = FunctionSpace(self.mesh, "DQ", 0)
