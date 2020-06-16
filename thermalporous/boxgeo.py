from firedrake import *

class BoxGeo():
    def __init__(self, Nx, Ny, Nz, params, Length = 365.76, Length_y = 365.76, Length_z = 1.8288, mg = False):
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.dim = 3
        self.params = params
        self.Length = Length # default is 365.76m, i.e. 1200 ft for SPE10
        self.Length_y = Length_y
        self.Length_z = Length_z
        if bool(mg):
            self.mesh = self.generate_mesh_hierarchy(Nx, Ny, Nz, mg["nref"])
        else:
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

    def generate_mesh_hierarchy(self, Nx, Ny, Nz, nref):
        Length = self.Length
        Length_y = self.Length_y
        Length_z = self.Length_z
        if (Nx % 2**nref == 0) is False:
            print("WARNING: Nx is not divisible by nref. Rounding up")
            nx = int(Nx/2**nref) + 1
            Nx = nx*2**nref
            self.Nx = Nx
        else:
            nx = int(Nx/2**nref)
        if (Ny % 2**nref == 0) is False:
            print("WARNING: Ny is not divisible by nref. Rounding up")
            ny = int(Ny/2**nref) + 1
            Ny = ny*2**nref
            self.Ny = Ny
        else:
            ny = int(Ny/2**nref)
        if (Nz % 2**nref == 0) is False:
            print("WARNING: Nz is not divisible by nref. Rounding up")
            nz = int(Nz/2**nref) + 1
            Nz = nz*2**nref
            self.Nz = Nz
        else:
            nz =int( Nz/2**nref)

        self.Dx = Length/Nx
        self.Dy = Length_y/Ny
        self.Dz = Length_z/Nz

        #distribution_parameters={"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)} # or FACET?
        meshbase = RectangleMesh(nx, ny, Length, Length_y, quadrilateral=True) #, distribution_parameters=distribution_parameters)
        base = MeshHierarchy(meshbase, nref)
        mh = ExtrudedMeshHierarchy(base, height = Length_z, base_layer = nz)
        self.mh = mh
        mesh = mh[-1]
        #from IPython import embed; embed()
        return mesh
        
    def init_function_space(self):
        self.V = FunctionSpace(self.mesh, "DQ", 0)
