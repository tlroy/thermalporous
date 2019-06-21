from firedrake import *
from thermalporous.boxgeo import BoxGeo

class HomogeneousBoxGeo(BoxGeo):
    def __init__(self, Nx, Ny, Nz, params, Length = 365.76, Length_y = 365.76, Length_z = 365.76):
        self.geotype = "Homogeneous"
        self.name = self.geotype + " " + str(Nx) + "X" + str(Ny) + "X" + str(Nz) + " grid"
        BoxGeo.__init__(self, Nx, Ny, Nz, params, Length = Length, Length_y = Length_y, Length_z = Length_z)
        
    def generate_geo_fields(self):
        # Create DG function for porosity field
        self.phi = Constant(0.2)

        # Create DG function for permeability field
        self.K = Constant(3E-7) # in mm^2

        # Define conductivity field
        self.kT = Constant(0.0)
        self.kT.assign(self.phi*self.params.ko + (1-self.phi)*self.params.kr) #We redefine this in the multiphase case such that it depends on saturation

