from data.create_SPE10_slice2D import create_SPE10_slice2D
from thermalporous.physicalparameters import PhysicalParameters as Params
from thermalporous.SPE10model import SPE10Model as GeoModel
from thermalporous.wellcase import WellCase as TestCase
from thermalporous.twophase import TwoPhase as ThermalModel

params = Params()
params.rate = 2e-4
params.S_o = 0.9

# Saving
savepvd = True


# TIME STEPPING
import sys
argument = sys.argv[3]
end = float(argument)
argument = sys.argv[2]
maxdt = float(argument)

# GEO
Nx = 60
Ny = 120
y_shift = 0
x_shift = 0

create_SPE10_slice2D(Nx, Ny, x_shift, y_shift)
geo = GeoModel(Nx, Ny, params, save = True)


# CASE
case = TestCase(params, geo, well_case = "SPE10_60x120") 

# MODEL
pcname = sys.argv[1]
parameters = "pc_" + pcname

model = ThermalModel(geo, case, params, end = end, maxdt = maxdt, save = savepvd, n_save = 1, small_dt_start = False, solver_parameters = parameters)

model.solve()
