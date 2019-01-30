from data.create_SPE10_slice2D import create_SPE10_slice2D
from thermalporous.physicalparameters import PhysicalParameters as Params
from thermalporous.SPE10model import SPE10Model as GeoModel
#from thermalporous.homogeneousgeo import HomogeneousGeo as GeoModel
from thermalporous.wellcase import WellCase as TestCase
from thermalporous.singlephase import SinglePhase as ThermalModel

params = Params()

# checkpointing and saving
savepvd = True
import os
suffix = os.path.splitext(__file__)[0]

initial = "_initial"
savename = suffix + "_initial_10d"
loadname = suffix + initial

checkpointing = {"save": False, "load": False, "loadname": loadname, "savename": savename}


# TIME STEPPING
import sys
argument = sys.argv[2]
end = float(argument)
end = end
maxdt = end
dt_init_fact = 2**(-7)
# GEO
Nx = 60
Ny = 120
y_shift = 0
x_shift = 0

create_SPE10_slice2D(Nx, Ny, x_shift, y_shift)


geo = GeoModel(Nx, Ny, params, save = False)

Length = geo.Length
Length_y = geo.Length_y

# CASE
case = TestCase(params, geo, well_case = "SPE10_60x120")

parameters = {
        "snes_type": "newtonls",
        "snes_monitor": True,
        "snes_converged_reason": True, 
        "snes_max_it": 15,

        "ksp_type": "fgmres",
        "ksp_converged_reason": True, 
        "ksp_view": False,
        "ksp_max_it": 200,
        "ksp_rtol": 1e-10,
        }

v_cycle = {"ksp_type": "preonly",
            "pc_type": "hypre",
            "pc_hypre_type" : "boomeramg",
            "pc_hypre_boomeramg_max_iter": 1,
            }   

pc_fieldsplit_cd = {"pc_type": "fieldsplit",      
        "pc_fieldsplit_type": "schur",
        #"pc_fieldsplit_schur_fact_type": "FULL",
        #"pc_fieldsplit_schur_fact_type": "diag",
        #"pc_fieldsplit_schur_fact_type": "upper",
        "pc_fieldsplit_schur_fact_type": "lower",

        "fieldsplit_0": v_cycle,    

        "fieldsplit_1_ksp_type": "preonly",
        "fieldsplit_1_pc_type": "python",
        "fieldsplit_1_pc_python_type": "thermalporous.preconditioners.ConvDiffSchurPC",
        "fieldsplit_1_schur": v_cycle,
            }


pc_fieldsplit_selfp = {"pc_type": "fieldsplit",      
        "pc_fieldsplit_type": "schur",
        "pc_fieldsplit_schur_fact_type": "FULL",
        "pc_fieldsplit_schur_precondition": "selfp",

        "fieldsplit_0": v_cycle,    
        "fieldsplit_1": v_cycle,
            }

pc_fieldsplit_a11 = {"pc_type": "fieldsplit",      
        "pc_fieldsplit_type": "schur",
        "pc_fieldsplit_schur_fact_type": "FULL",
        "pc_fieldsplit_schur_precondition": "a11",

        "fieldsplit_0": v_cycle,    
        "fieldsplit_1": v_cycle,
            }


pc_cpr = {"pc_type": "composite",
        "pc_composite_type": "multiplicative",
        "pc_composite_pcs": "python,bjacobi",

        "sub_0_pc_python_type": "thermalporous.preconditioners.CPRStage1PC",
        "sub_0_cpr_stage1": v_cycle,

        "sub_1_pc_bjacobi_blocks": 1,
        "sub_1_sub_pc_type": "ilu",
        "sub_1_sub_pc_factor_levels": 0,
        "mat_type": "aij",
        }

pc_cpr1 = {"pc_type": "composite",
        "pc_composite_type": "multiplicative",
        "pc_composite_pcs": "python,bjacobi",

        "sub_0_pc_python_type": "thermalporous.preconditioners.CPRStage1PC",
        "sub_0_cpr_stage1": v_cycle,

        "sub_1_pc_bjacobi_blocks": 1,
        "sub_1_sub_pc_type": "ilu",
        "sub_1_sub_pc_factor_levels": 1,
        "mat_type": "aij",
        }

pc_cpr_gmres = {"pc_type": "composite",
        "pc_composite_type": "multiplicative",
        "pc_composite_pcs": "fieldsplit,bjacobi",
        
        "sub_0_pc_fieldsplit_type": "additive",
        "sub_0_fieldsplit_0": v_cycle,    
        "sub_0_fieldsplit_1_ksp_type": "gmres",
        "sub_0_fieldsplit_1_ksp_max_it": 0,
        "sub_0_fieldsplit_1_pc_type": "none",

        #"sub_1_pc_bjacobi_blocks": 1,
        "sub_1_sub_pc_type": "ilu",
        "sub_1_sub_pc_factor_levels": 0,
        "mat_type": "aij",
        }


pcname = sys.argv[1]
if pcname == "fieldsplit_cd"    : pc = pc_fieldsplit_cd
if pcname == "fieldsplit_a11"   : pc = pc_fieldsplit_a11
if pcname == "fieldsplit_selfp" : pc = pc_fieldsplit_selfp

if pcname == "cprlu"            : pc = pc_cprlu 
if pcname == "cpr"              : pc = pc_cpr
if pcname == "cpr1"             : pc = pc_cpr1
if pcname == "cpr_gmres"	: pc = pc_cpr_gmres

parameters.update(pc)

resultsname = suffix + "_" + pcname + "_results.txt"

model = ThermalModel(geo, case, params, end = end, maxdt = maxdt, save = savepvd, n_save = 10, small_dt_start = True, checkpointing = checkpointing,  solver_parameters = parameters, filename = resultsname, dt_init_fact = dt_init_fact)
model.solve()

lits = model.total_lits
nits = model.total_nits


filename = suffix + "_" + pcname + initial + "_iterations.txt"
f = open(filename, "a")
print(end, lits, nits, file = f)
