from thermalporous.physicalparameters import PhysicalParameters as Params
from thermalporous.homogeneousgeo import HomogeneousGeo as GeoModel
from thermalporous.wellcase import WellCase as TestCase
from thermalporous.singlephase import SinglePhase as ThermalModel

params = Params()
#params.p_inj = 0.1*params.p_inj
#params.rate = 0.1*params.rate
#params.rate = 5.787037e-5
params.rate = 1e-6
#params.T_inj = 373.15
params.T_prod = 320.0


# checkpointing and saving
savepvd = True
import os
suffix = os.path.splitext(__file__)[0]

initial = "_initial_500d" 
savename = suffix + "_initial_100d" 
loadname = suffix + initial

checkpointing = {"save": False, "load": False, "loadname": loadname, "savename": savename}


# TIME STEPPING
import sys
argument = sys.argv[2]
dt = float(argument)
end = 2*dt
maxdt = dt
#maxdt = 10

# GEO
N = sys.argv[3]
N = int(N)
Nx = N
Ny = N
L = 20.

geo = GeoModel(Nx, Ny, params, L, L)
Length = geo.Length
Length_y = geo.Length_y
geo.K_x = geo.K_x
geo.K_y = geo.K_y

# CASE
prod_points = [[Length/4, Length_y/2]]
inj_points = [[3*Length/4, Length_y/2]]
case = TestCase(params, geo, well_case = "test0", constant_rate = True)#, prod_points = prod_points, inj_points = inj_points)
#case = TestCase(params, geo, prod_points = [], inj_points = [], constant_rate = True)

parameters = {
        "snes_type": "newtonls",
        "snes_monitor": None,
        "snes_converged_reason": None, 
        "snes_max_it": 15,

        "ksp_type": "fgmres",
        "ksp_converged_reason": None, 
        #"ksp_view": None,
        "ksp_max_it": 200,
        #"ksp_view": True,
        }

v_cycle = {"ksp_type": "preonly",
            "pc_type": "hypre",
            "pc_hypre_type" : "boomeramg",
            "pc_hypre_boomeramg_max_iter": 1,
            }   

pc_fieldsplit_cd = {"pc_type": "fieldsplit",      
        "pc_fieldsplit_type": "schur",
        "pc_fieldsplit_schur_fact_type": "FULL",

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

pc_cpr_bilu = {"pc_type": "composite",
        "pc_composite_type": "multiplicative",
        "pc_composite_pcs": "python,bjacobi",

        "sub_0_pc_python_type": "thermalporous.preconditioners.CPRStage1PC",
        "sub_0_cpr_stage1": v_cycle,

        "sub_1_pc_bjacobi_blocks": 16,
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
if pcname == "cpr_bilu"         : pc = pc_cpr_bilu
if pcname == "cpr_gmres"         : pc = pc_cpr_gmres 


parameters.update(pc)

resultsname = suffix + "_" + pcname + "_results.txt"

model = ThermalModel(geo, case, params, end = end, maxdt = maxdt, save = savepvd, n_save = 1, small_dt_start = False, checkpointing = checkpointing,  solver_parameters = parameters, filename = resultsname)

model.solve()

#lits = model.total_lits
#nits = model.total_nits


#filename = suffix + "_" + pcname + initial + "_iterations.txt"
#f = open(filename, "a")
#print(end, lits, nits, file = f)
