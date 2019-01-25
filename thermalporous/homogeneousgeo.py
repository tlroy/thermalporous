from firedrake import *
from thermalporous.rectanglegeo import RectangleGeo

class HomogeneousGeo(RectangleGeo):
    def __init__(self, Nx, Ny, params, Length, Length_y):
        self.geotype = "Homogeneous"
        self.name = self.geotype + " " + str(Nx) + "X" + str(Ny) + " grid"
        RectangleGeo.__init__(self, Nx, Ny, params, Length, Length_y)
        
    def generate_geo_fields(self):
        # Create DG function for porosity field
        self.phi = Constant(0.2)

        # Create DG function for permeability field
        self.K = Constant(3E-7) # in mm^2

        # Define conductivity field
        self.kT = Constant(0.0)
        self.kT.assign(self.phi*self.params.ko + (1-self.phi)*self.params.kr)
        
