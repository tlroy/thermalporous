from thermalporous.physicalparameters import PhysicalParameters as Params
from thermalporous.homogeneousboxgeo import HomogeneousBoxGeo as GeoModel
from thermalporous.heatercase import HeaterCase as TestCase
from thermalporous.twophase import TwoPhase as ThermalModel
import sys
params = Params()
#params.p_inj = 0.1*params.p_inj
#params.rate = 0.1*params.rate
#params.rate = 5.787037e-5
#params.rate = 1e-6
params.rate = 1e-7
params.T_inj = 373.15
#params.T_prod = 320.0

argument = sys.argv[4]
S_o = float(argument)
params.S_o = S_o

# checkpointing and saving
savepvd = False
import os
suffix = os.path.splitext(__file__)[0]

n_proc = sys.argv[3]

initial = "_initial_1d" + "_np" + str(n_proc)
savename = suffix + initial
loadname = suffix + initial

init_save = 2
#init_save = int(sys.argv[4])
if init_save == 1:
    checkpointing = {"save": True, "load": False, "loadname": loadname, "savename": savename}
elif init_save == 0:
    checkpointing = {"save": False, "load": True, "loadname": loadname, "savename": savename}
else:
    checkpointing = {"save": False, "load": False, "loadname": loadname, "savename": savename}


# TIME STEPPING

argument = sys.argv[2]
dt = float(argument)
if init_save == 1:
    end = 1*dt
else:
    end = 5*dt
maxdt = dt
#maxdt = 10


nproc = float(n_proc)

dofs_per_p = 5e4

total_dofs = dofs_per_p*nproc

import numpy as np
N = np.ceil( (total_dofs/3.0)**(1./3.) )

Nx = N
Ny = N
Nz = N


L = 50
Length = 50
Length_y = 50
Length_z = 50


geo = GeoModel(Nx, Ny, Nz, params, Length = Length, Length_y = Length_y, Length_z = Length_z)
fact = 1e0
geo.K_x = geo.K_x*fact
geo.K_y = geo.K_y*fact
geo.K_y = geo.K_y*fact
Length = geo.Length
Length_y = geo.Length_y
Length_z = geo.Length_z

# CASE
prod_points = [[Length/8, Length_y/2, Length_z*0.2], [Length/4, Length_y/2, Length_z*0.2], [3*Length/8, Length_y/2, Length_z*0.2], [Length/2, Length_y/2, Length_z*0.2], [5*Length/8, Length_y/2, Length_z*0.2], [3*Length/4, Length_y/2, Length_z*0.2], [7*Length/8, Length_y/2, Length_z*0.2]] + [[Length/8, Length_y/4, Length_z*0.2], [Length/4, Length_y/4, Length_z*0.2], [3*Length/8, Length_y/4, Length_z*0.2], [Length/2, Length_y/4, Length_z*0.2], [5*Length/8, Length_y/4, Length_z*0.2], [3*Length/4, Length_y/4, Length_z*0.2], [7*Length/8, Length_y/4, Length_z*0.2]] + [[Length/8, 3*Length_y/4, Length_z*0.2], [Length/4, 3*Length_y/4, Length_z*0.2], [3*Length/8, 3*Length_y/4, Length_z*0.2], [Length/2, 3*Length_y/4, Length_z*0.2], [5*Length/8, 3*Length_y/4, Length_z*0.2], [3*Length/4, 3*Length_y/4, Length_z*0.2], [7*Length/8, Length_y/4, Length_z*0.2]]
inj_points = [[Length/8, Length_y/2, Length_z*0.8], [Length/4, Length_y/2, Length_z*0.8], [3*Length/8, Length_y/2, Length_z*0.8], [Length/2, Length_y/2, Length_z*0.8], [5*Length/8, Length_y/2, Length_z*0.8], [3*Length/4, Length_y/2, Length_z*0.8], [7*Length/8, Length_y/2, Length_z*0.8]] + [[Length/8, Length_y/4, Length_z*0.8], [Length/4, Length_y/4, Length_z*0.8], [3*Length/8, Length_y/4, Length_z*0.8], [Length/2, Length_y/4, Length_z*0.8], [5*Length/8, Length_y/4, Length_z*0.8], [3*Length/4, Length_y/4, Length_z*0.8], [7*Length/8, Length_y/4, Length_z*0.8]] + [[Length/8, 3*Length_y/4, Length_z*0.8], [Length/4, 3*Length_y/4, Length_z*0.8], [3*Length/8, 3*Length_y/4, Length_z*0.8], [Length/2, 3*Length_y/4, Length_z*0.8], [5*Length/8, 3*Length_y/4, Length_z*0.8], [3*Length/4, 3*Length_y/4, Length_z*0.8], [7*Length/8, Length_y/4, Length_z*0.8]]
heater_points = prod_points + inj_points
case = TestCase(params, geo, well_case = None, heater_points = heater_points)#, constant_rate = True)

parameters = {
        "snes_type": "newtonls",
        "snes_monitor": None,
        "snes_converged_reason": None, 
        "snes_max_it": 15,

        "ksp_type": "fgmres",
        "ksp_converged_reason": None, 
        #"ksp_view": None,
        "ksp_max_it": 200,

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

        "sub_0_pc_python_type": "thermalporous.preconditioners.CPTRStage1_originalPC",
        "sub_0_cpr_stage1_pc_type": "fieldsplit",
        "sub_0_cpr_stage1_pc_fieldsplit_type": "schur",
        "sub_0_cpr_stage1_pc_fieldsplit_schur_fact_type": "FULL",
        
        "sub_0_cpr_stage1_fieldsplit_1_ksp_type": "preonly",
        "sub_0_cpr_stage1_fieldsplit_1_pc_type": "python",
        "sub_0_cpr_stage1_fieldsplit_1_pc_python_type": "thermalporous.preconditioners.ConvDiffSchurTwoPhasesPC",
        "sub_0_cpr_stage1_fieldsplit_1_schur": v_cycle,

        "sub_0_cpr_stage1_fieldsplit_0": v_cycle,
        
        "sub_1_pc_bjacobi_blocks": 1,
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

        "sub_1_pc_bjacobi_blocks": 1,
        "sub_1_sub_pc_type": "ilu",
        #"sub_1_sub_pc_type": "jacobi",
        "sub_1_sub_pc_factor_levels": 0,
        "mat_type": "aij",
        }

pc_cptra11 = {"pc_type": "composite",
        "pc_composite_type": "multiplicative",
        "pc_composite_pcs": "python,bjacobi",

        "sub_0_pc_python_type": "thermalporous.preconditioners.CPTRStage1_originalPC",
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
        "sub_0_fieldsplit_0": v_cycle,    
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

pc_fieldsplit = {"pc_type": "composite",
                "pc_composite_type": "multiplicative",
                "pc_composite_pcs": "fieldsplit,bjacobi",
                 
                "sub_0_pc_fieldsplit_0_fields": "0,1",
                "sub_0_pc_fieldsplit_1_fields": "2",
                "sub_0_pc_fieldsplit_type": "multiplicative",
                #"sub_0_pc_fieldsplit_type": "schur",
                #"sub_0_pc_fieldsplit_schur_fact_type": "FULL",
               
                "sub_0_fieldsplit_0_ksp_type": "preonly",
                #"sub_0_fieldsplit_0_pc_type": "lu",
                "sub_0_fieldsplit_0_pc_type": "fieldsplit", 
                "sub_0_fieldsplit_0_pc_fieldsplit_type": "schur",
                "sub_0_fieldsplit_0_pc_fieldsplit_schur_fact_type": "FULL",
                 
                "sub_0_fieldsplit_0_fieldsplit_1_ksp_type": "preonly",
                "sub_0_fieldsplit_0_fieldsplit_1_pc_type": "python",
                "sub_0_fieldsplit_0_fieldsplit_1_pc_python_type": "thermalporous.preconditioners.ConvDiffSchurTwoPhasesPC",
                "sub_0_fieldsplit_0_fieldsplit_1_schur": v_cycle,
                 
                "sub_0_fieldsplit_0_fieldsplit_0": v_cycle,       
                
                #"sub_0_pc_fieldsplit_schur_precondition": "a11",
                #"sub_0_fieldsplit_1_ksp_type": "preonly",
                #"sub_0_fieldsplit_1_pc_type": "lu",
                "sub_0_fieldsplit_1_ksp_type": "gmres",
                "sub_0_fieldsplit_1_ksp_max_it": 0,
                "sub_0_fieldsplit_1_pc_type": "none",
                
                
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

pc_bilu = {"pc_type": "bjacobi",
           "sub_pc_type": "ilu",
           "mat_type": "aij",}

pcname = sys.argv[1]

if pcname == "cprlu"            : pc = pc_cprlu 
if pcname == "cpr"              : pc = pc_cpr
if pcname == "cpr1"             : pc = pc_cpr1
if pcname == "cpr_bilu"         : pc = pc_cpr_bilu
if pcname == "cpr_gmres"        : pc = pc_cpr_gmres 
if pcname == "cpr2_gmres"       : pc = pc_cpr2_gmres
if pcname == "cprilu1_gmres"    : pc = pc_cprilu1_gmres

if pcname == "cptr"             : pc = pc_cptr
if pcname == "cptr_gmres"       : pc = pc_cptr_gmres
if pcname == "cptr2_gmres"      : pc = pc_cptr2_gmres

if pcname == "fieldsplit"       : pc = pc_fieldsplit

if pcname == "bilu"             : pc = pc_bilu

#parameters.update(pc)

parameters = "pc_" + pcname

resultsname = suffix + "_" + pcname + "_results.txt"

model = ThermalModel(geo, case, params, end = end, maxdt = maxdt, save = savepvd, n_save = 1, small_dt_start = False, checkpointing = checkpointing,  solver_parameters = parameters, filename = resultsname)

model.solve()

#(pvec, Tvec, S_ovec) = model.u.split()

#from IPython import embed; embed()

#lits = model.total_lits
#nits = model.total_nits


#filename = suffix + "_" + pcname + initial + "_iterations.txt"
#f = open(filename, "a")
#print(end, lits, nits, file = f)
