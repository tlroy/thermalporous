from thermalporous.physicalparameters import PhysicalParameters as Params
from thermalporous.homogeneousgeo import HomogeneousGeo as GeoModel
from thermalporous.wellcase import WellCase as TestCase
from thermalporous.twophase import TwoPhase as ThermalModel

params = Params()
#params.p_inj = 0.1*params.p_inj
#params.rate = 0.1*params.rate
#params.rate = 5.787037e-5
#params.rate = 1e-6
params.rate = 5e-8
params.T_inj = 373.15
#params.T_prod = 320.0

params.S_o = 0.9


# checkpointing and saving
savepvd = False
import os
suffix = os.path.splitext(__file__)[0]

initial = "_initial_smalld" 
savename = suffix + "_initial_smalld" 
loadname = suffix + initial

checkpointing = {"save": False, "load": False, "loadname": loadname, "savename": savename}


# TIME STEPPING
import sys
argument = sys.argv[2]
dt = float(argument)
end = 5*dt
maxdt = dt
#maxdt = 10

# GEO
N = sys.argv[3]
N = float(N)
Nx = N
Ny = N
L = 20

geo = GeoModel(Nx, Ny, params, L, L)
Length = geo.Length
Length_y = geo.Length_y
geo.K_x = geo.K_x
geo.K_y = geo.K_y

# CASE
case = TestCase(params, geo, well_case = "test0", constant_rate = True)#, prod_points = prod_points, inj_points = inj_points)

parameters = {
        "snes_type": "newtonls",
        "snes_monitor": None,
        "snes_converged_reason": None, 
        "snes_max_it": 15,

        "ksp_type": "fgmres",
        "ksp_converged_reason": None, 
        #"ksp_view": None,
        "ksp_max_it": 200,
        #"ksp_view": None,
        "ksp_gmres_restart": 200,
        "ksp_rtol": 1e-8,
        }

v_cycle = {"ksp_type": "preonly",
            "pc_type": "hypre",
            "pc_hypre_type" : "boomeramg",
            "pc_hypre_boomeramg_max_iter": 1,
            }   

lu = {"ksp_type": "preonly",
      "pc_type": "lu",}

pc_cptr = {"pc_type": "composite",
        "pc_composite_type": "multiplicative",
        "pc_composite_pcs": "python,bjacobi",

        "sub_0_pc_python_type": "thermalporous.preconditioners.CPTRStage1PC",
        "sub_0_cpr_stage1_pc_type": "fieldsplit",
        "sub_0_cpr_stage1_pc_fieldsplit_type": "schur",
        "sub_0_cpr_stage1_pc_fieldsplit_schur_fact_type": "FULL",
        
        "sub_0_cpr_stage1_fieldsplit_1_ksp_type": "preonly",
        "sub_0_cpr_stage1_fieldsplit_1_pc_type": "python",
        "sub_0_cpr_stage1_fieldsplit_1_pc_python_type": "thermalporous.preconditioners.ConvDiffSchurTwoPhasesPC",
        
        "sub_0_cpr_stage1_fieldsplit_1_schur": v_cycle,

        "sub_0_cpr_stage1_fieldsplit_0": v_cycle,
        
        "sub_1_sub_pc_type": "ilu",
        #"sub_1_sub_pc_type": "jacobi",
        "sub_1_sub_pc_factor_levels": 0,
        "mat_type": "aij",
        }

pc_cpr = {"pc_type": "composite",
        "pc_composite_type": "multiplicative",
        "pc_composite_pcs": "python,bjacobi",

        "sub_0_pc_python_type": "thermalporous.preconditioners.CPRStage1PC",
        "sub_0_cpr_stage1": v_cycle,

        "sub_1_sub_pc_type": "ilu",
        #"sub_1_sub_pc_type": "jacobi",
        "sub_1_sub_pc_factor_levels": 0,
        "mat_type": "aij",
        }



pc_cptra11 = {"pc_type": "composite",
        "pc_composite_type": "multiplicative",
        "pc_composite_pcs": "python,bjacobi",

        "sub_0_pc_python_type": "thermalporous.preconditioners.CPTRStage1PC",
        "sub_0_cpr_stage1_pc_type": "fieldsplit",
        "sub_0_cpr_stage1_pc_fieldsplit_type": "schur",
        "sub_0_cpr_stage1_pc_fieldsplit_schur_fact_type": "FULL",
        
        "sub_0_cpr_stage1_pc_fieldsplit_schur_precondition": "a11",
        #"sub_0_cpr_stage1_pc_fieldsplit_schur_precondition": "selfp",
        "sub_0_cpr_stage1_fieldsplit_1": v_cycle,

        "sub_0_cpr_stage1_fieldsplit_0": v_cycle,
        
        "sub_1_pc_bjacobi_blocks": 1,
        "sub_1_sub_pc_type": "ilu",
        "sub_1_sub_pc_factor_levels": 0,
        "mat_type": "aij",
        }

pc_cpr_gmres = {"pc_type": "composite",
        "pc_composite_type": "multiplicative",
        "pc_composite_pcs": "fieldsplit,bjacobi",
        
        "sub_0_pc_fieldsplit_0_fields": "0",
        "sub_0_pc_fieldsplit_1_fields": "1,2",
        "sub_0_pc_fieldsplit_type": "additive",
        "sub_0_fieldsplit_0": lu,    
        "sub_0_fieldsplit_1_ksp_type": "gmres",
        "sub_0_fieldsplit_1_ksp_max_it": 0,
        "sub_0_fieldsplit_1_pc_type": "none",

        #"sub_1_pc_bjacobi_blocks": 1,
        "sub_1_sub_pc_type": "ilu",
        "sub_1_sub_pc_factor_levels": 0,
        "mat_type": "aij",
        }

pc_cprilu1_gmres = {"pc_type": "composite",
        "pc_composite_type": "multiplicative",
        "pc_composite_pcs": "fieldsplit,bjacobi",
        
        "sub_0_pc_fieldsplit_0_fields": "0",
        "sub_0_pc_fieldsplit_1_fields": "1,2",
        "sub_0_pc_fieldsplit_type": "additive",
        "sub_0_fieldsplit_0": v_cycle,    
        "sub_0_fieldsplit_1_ksp_type": "gmres",
        "sub_0_fieldsplit_1_ksp_max_it": 0,
        "sub_0_fieldsplit_1_pc_type": "none",

        #"sub_1_pc_bjacobi_blocks": 1,
        "sub_1_sub_pc_type": "ilu",
        "sub_1_sub_pc_factor_levels": 1,
        "mat_type": "aij",
        }

pc_cptr_gmres = {"pc_type": "composite",
        "pc_composite_type": "multiplicative",
        "pc_composite_pcs": "fieldsplit,bjacobi",
        
        "sub_0_pc_fieldsplit_0_fields": "0,1",
        "sub_0_pc_fieldsplit_1_fields": "2",
        "sub_0_pc_fieldsplit_type": "additive",
        
        "sub_0_fieldsplit_0_pc_type": "fieldsplit", 
        "sub_0_fieldsplit_0_pc_fieldsplit_type": "schur",
        "sub_0_fieldsplit_0_pc_fieldsplit_schur_fact_type": "FULL",
  
        "sub_0_fieldsplit_0_fieldsplit_1_ksp_type": "preonly",
        "sub_0_fieldsplit_0_fieldsplit_1_pc_type": "python",
        "sub_0_fieldsplit_0_fieldsplit_1_pc_python_type": "thermalporous.preconditioners.ConvDiffSchurTwoPhasesPC",
        "sub_0_fieldsplit_0_fieldsplit_1_schur": v_cycle,
        
        "sub_0_fieldsplit_0_fieldsplit_0": v_cycle,       
        
        "sub_0_fieldsplit_1_ksp_type": "gmres",
        "sub_0_fieldsplit_1_ksp_max_it": 0,
        "sub_0_fieldsplit_1_pc_type": "none",

        #"sub_1_pc_bjacobi_blocks": 1,
        "sub_1_sub_pc_type": "ilu",
        "sub_1_sub_pc_factor_levels": 0,
        "mat_type": "aij",
        }

pc_cptr2_gmres = {"pc_type": "composite",
        "pc_composite_type": "multiplicative",
        "pc_composite_pcs": "bjacobi,fieldsplit",
        
        "sub_0_sub_pc_type": "ilu",
        "sub_0_sub_pc_factor_levels": 0,
        
        "sub_1_pc_fieldsplit_0_fields": "0,1",
        "sub_1_pc_fieldsplit_1_fields": "2",
        "sub_1_pc_fieldsplit_type": "additive",
        
        #"sub_1_fieldsplit_0_pc_type": "lu", 
             
        "sub_1_fieldsplit_0_pc_type": "fieldsplit", 
        "sub_1_fieldsplit_0_pc_fieldsplit_type": "schur",
        "sub_1_fieldsplit_0_pc_fieldsplit_schur_fact_type": "FULL",
             
        "sub_1_fieldsplit_0_fieldsplit_1_ksp_type": "preonly",
        "sub_1_fieldsplit_0_fieldsplit_1_pc_type": "python",
        "sub_1_fieldsplit_0_fieldsplit_1_pc_python_type": "thermalporous.preconditioners.ConvDiffSchurTwoPhasesPC",
        "sub_1_fieldsplit_0_fieldsplit_1_schur": v_cycle,
             
        "sub_1_fieldsplit_0_fieldsplit_0": v_cycle,       
             
        "sub_1_fieldsplit_1_ksp_type": "gmres",
        "sub_1_fieldsplit_1_ksp_max_it": 0,
        "sub_1_fieldsplit_1_pc_type": "none",

        "mat_type": "aij",
        }

pc_cpr2_gmres = {"pc_type": "composite",
        "pc_composite_type": "multiplicative",
        "pc_composite_pcs": "bjacobi,fieldsplit",
        
        "sub_0_sub_pc_type": "ilu",
        "sub_0_sub_pc_factor_levels": 0,
        
        "sub_1_pc_fieldsplit_0_fields": "0",
        "sub_1_pc_fieldsplit_1_fields": "1,2",
        "sub_1_pc_fieldsplit_type": "additive",
        "sub_1_fieldsplit_0": v_cycle,    
        "sub_1_fieldsplit_1_ksp_type": "gmres",
        "sub_1_fieldsplit_1_ksp_max_it": 0,
        "sub_1_fieldsplit_1_pc_type": "none",

        "mat_type": "aij",
        }

pc_cptr4_gmres = {"pc_type": "composite",
        "pc_composite_type": "multiplicative",
        "pc_composite_pcs": "bjacobi,fieldsplit,bjacobi,",
        
        "sub_0_sub_pc_type": "ilu",
        "sub_0_sub_pc_factor_levels": 0,
        
        "sub_1_pc_fieldsplit_0_fields": "0,1",
        "sub_1_pc_fieldsplit_1_fields": "2",
        "sub_1_pc_fieldsplit_type": "additive",
             
        "sub_1_fieldsplit_0_pc_type": "fieldsplit", 
        "sub_1_fieldsplit_0_pc_fieldsplit_type": "schur",
        "sub_1_fieldsplit_0_pc_fieldsplit_schur_fact_type": "FULL",
             
        "sub_1_fieldsplit_0_fieldsplit_1_ksp_type": "preonly",
        "sub_1_fieldsplit_0_fieldsplit_1_pc_type": "python",
        "sub_1_fieldsplit_0_fieldsplit_1_pc_python_type": "thermalporous.preconditioners.ConvDiffSchurTwoPhasesPC",
        "sub_1_fieldsplit_0_fieldsplit_1_schur": v_cycle,
             
        "sub_1_fieldsplit_0_fieldsplit_0": v_cycle,       
             
        "sub_1_fieldsplit_1_ksp_type": "gmres",
        "sub_1_fieldsplit_1_ksp_max_it": 0,
        "sub_1_fieldsplit_1_pc_type": "none",
        
        "sub_2_sub_pc_type": "ilu",
        "sub_2_sub_pc_factor_levels": 0,

        "mat_type": "aij",
        }

pc_cpr4_gmres = {"pc_type": "composite",
        "pc_composite_type": "multiplicative",
        "pc_composite_pcs": "bjacobi,fieldsplit,bjacobi,",
        
        "sub_0_sub_pc_type": "ilu",
        "sub_0_sub_pc_factor_levels": 0,
        
        "sub_1_pc_fieldsplit_0_fields": "0",
        "sub_1_pc_fieldsplit_1_fields": "1,2",
        "sub_1_pc_fieldsplit_type": "additive",
        "sub_1_fieldsplit_0": v_cycle,    
        "sub_1_fieldsplit_1_ksp_type": "gmres",
        "sub_1_fieldsplit_1_ksp_max_it": 0,
        "sub_1_fieldsplit_1_pc_type": "none",
        
        "sub_2_sub_pc_type": "ilu",
        "sub_2_sub_pc_factor_levels": 0,

        "mat_type": "aij",
        }



pc_cptr3_gmres = {"pc_type": "composite",
        "pc_composite_type": "multiplicative",
        "pc_composite_pcs": "fieldsplit,bjacobi,fieldsplit",
        
        "sub_0_pc_fieldsplit_0_fields": "0,1",
        "sub_0_pc_fieldsplit_1_fields": "2",
        "sub_0_pc_fieldsplit_type": "additive",
             
        "sub_0_fieldsplit_0_pc_type": "fieldsplit", 
        "sub_0_fieldsplit_0_pc_fieldsplit_type": "schur",
        "sub_0_fieldsplit_0_pc_fieldsplit_schur_fact_type": "FULL",
             
        "sub_0_fieldsplit_0_fieldsplit_1_ksp_type": "preonly",
        "sub_0_fieldsplit_0_fieldsplit_1_pc_type": "python",
        "sub_0_fieldsplit_0_fieldsplit_1_pc_python_type": "thermalporous.preconditioners.ConvDiffSchurTwoPhasesPC",
        "sub_0_fieldsplit_0_fieldsplit_1_schur": v_cycle,
             
        "sub_0_fieldsplit_0_fieldsplit_0": v_cycle,       
             
        "sub_0_fieldsplit_1_ksp_type": "gmres",
        "sub_0_fieldsplit_1_ksp_max_it": 0,
        "sub_0_fieldsplit_1_pc_type": "none",
        
        "sub_1_sub_pc_type": "ilu",
        "sub_1_sub_pc_factor_levels": 0,
        
        "sub_2_pc_fieldsplit_0_fields": "0,1",
        "sub_2_pc_fieldsplit_1_fields": "2",
        "sub_2_pc_fieldsplit_type": "additive",
             
        "sub_2_fieldsplit_0_pc_type": "fieldsplit", 
        "sub_2_fieldsplit_0_pc_fieldsplit_type": "schur",
        "sub_2_fieldsplit_0_pc_fieldsplit_schur_fact_type": "FULL",
             
        "sub_2_fieldsplit_0_fieldsplit_1_ksp_type": "preonly",
        "sub_2_fieldsplit_0_fieldsplit_1_pc_type": "python",
        "sub_2_fieldsplit_0_fieldsplit_1_pc_python_type": "thermalporous.preconditioners.ConvDiffSchurTwoPhasesPC",
        "sub_2_fieldsplit_0_fieldsplit_1_schur": v_cycle,
             
        "sub_2_fieldsplit_0_fieldsplit_0": v_cycle,       
             
        "sub_2_fieldsplit_1_ksp_type": "gmres",
        "sub_2_fieldsplit_1_ksp_max_it": 0,
        "sub_2_fieldsplit_1_pc_type": "none",

        "mat_type": "aij",
        }

pc_cpr3_gmres = {"pc_type": "composite",
        "pc_composite_type": "multiplicative",
        "pc_composite_pcs": "fieldsplit,bjacobi,fieldsplit",
        
        "sub_0_pc_fieldsplit_0_fields": "0",
        "sub_0_pc_fieldsplit_1_fields": "1,2",
        "sub_0_pc_fieldsplit_type": "additive",
        "sub_0_fieldsplit_0": v_cycle,    
        "sub_0_fieldsplit_1_ksp_type": "gmres",
        "sub_0_fieldsplit_1_ksp_max_it": 0,
        "sub_0_fieldsplit_1_pc_type": "none",
        
        "sub_1_sub_pc_type": "ilu",
        "sub_1_sub_pc_factor_levels": 0,
        
        "sub_2_pc_fieldsplit_0_fields": "0",
        "sub_2_pc_fieldsplit_1_fields": "1,2",
        "sub_2_pc_fieldsplit_type": "additive",
        "sub_2_fieldsplit_0": v_cycle,    
        "sub_2_fieldsplit_1_ksp_type": "gmres",
        "sub_2_fieldsplit_1_ksp_max_it": 0,
        "sub_2_fieldsplit_1_pc_type": "none",

        "mat_type": "aij",
        }

pc_hypre = {"pc_type": "hypre",
          "pc_hypre_type": "boomeramg",
          #"mat_type": "aij",
          }

pc_fieldsplit_diag = {"pc_type": "fieldsplit",
                              "pc_fieldsplit_type": "additive",
                              "fieldsplit_0": v_cycle,
                              "fieldsplit_1": v_cycle,
                              }

pc_cptramg_gmres = {"pc_type": "composite",
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



pc_bilu = {"pc_type": "bjacobi",
           "sub_pc_type": "ilu",
           "mat_type": "aij",}

pc_lu = lu
pc_lu.update({"mat_type": "aij"})

pcname = sys.argv[1]
vector = False
if pcname == "cprlu"            : pc = pc_cprlu 
if pcname == "cpr"              : pc = pc_cpr
if pcname == "cpr1"             : pc = pc_cpr1
if pcname == "cpr_bilu"         : pc = pc_cpr_bilu
if pcname == "cpr_gmres"        : pc = pc_cpr_gmres 
if pcname == "cpr2_gmres"       : pc = pc_cpr2_gmres
if pcname == "cpr3_gmres"       : pc = pc_cpr3_gmres
if pcname == "cpr4_gmres"       : pc = pc_cpr4_gmres
if pcname == "cprilu1_gmres"    : pc = pc_cprilu1_gmres

if pcname == "cptr"             : pc = pc_cptr
if pcname == "cptr_gmres"       : pc = pc_cptr_gmres
if pcname == "cptr2_gmres"      : pc = pc_cptr2_gmres
if pcname == "cptr3_gmres"      : pc = pc_cptr3_gmres
if pcname == "cptr4_gmres"      : pc = pc_cptr4_gmres

if pcname == "bilu"             : pc = pc_bilu
if pcname == "lu"               : pc = pc_lu

if pcname == "hypre"            : pc = pc_hypre; vector = True
if pcname == "fs_diag"          : pc = pc_fieldsplit_diag; vector = True
if pcname == "cptramg_gmres"    : pc = pc_cptramg_gmres; vector = True

#parameters.update(pc)

parameters = "pc_" + pcname # using default options instead

#parameters.update({"ksp_monitor_residuals": None,})

resultsname = suffix + "_" + pcname + "_results.txt"

model = ThermalModel(geo, case, params, end = end, maxdt = maxdt, save = savepvd, n_save = 1, small_dt_start = False, checkpointing = checkpointing,  solver_parameters = parameters, filename = resultsname, vector = vector)

model.solve()

#(pvec, Tvec, S_ovec) = model.u.split()

#from IPython import embed; embed()

#lits = model.total_lits
#nits = model.total_nits


#filename = suffix + "_" + pcname + initial + "_iterations.txt"
#f = open(filename, "a")
#print(end, lits, nits, file = f)
