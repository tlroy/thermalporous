import numpy as np

from firedrake import *
from thermalporous.thermalmodel import ThermalModel
from thermalporous.preconditioners import CPTRStage1, ConvDiffSchurTwoPhasesPC, CPTRStage1PC, CPTRStage1_originalPC

from firedrake.utils import cached_property
class TwoPhase(ThermalModel):
    def __init__(self, geo, case, params, end = 1.0, maxdt = 0.005, save = False, n_save = 2, small_dt_start = True, checkpointing = {}, solver_parameters = None, filename = "results/results.txt", dt_init_fact = 2**(-10)):
        self.name = "Two-phase"
        self.geo = geo
        self.case = case
        self.params = params
        self.mesh = geo.mesh
        self.comm = self.mesh.comm
        self.V = geo.V
        self.W = self.V*self.V*self.V
        self.save = save
        self.n_save = n_save
        self.small_dt_start = small_dt_start
        self.solver_parameters = solver_parameters
        if self.geo.dim == 2:
            self.init_variational_form = self.init_variational_form_2D
        elif self.geo.dim == 3:
            self.init_variational_form = self.init_variational_form_3D

        try:
            self.case.prod_wells
        except AttributeError:
            self.case.prod_wells = list()
        try:
            self.case.inj_wells
        except AttributeError:
            self.case.inj_wells = list()  
        try:
            self.case.heaters
        except AttributeError:
            self.case.heaters = list()  
        try:
            self.bcs = case.init_bcs()
        except AttributeError:
            self.bcs = []
        
        
        ThermalModel.__init__(self, end = end, maxdt = maxdt, save = save, n_save = n_save, small_dt_start = small_dt_start, checkpointing = checkpointing, filename = filename, dt_init_fact = dt_init_fact)
        
    def init_IC_uniform(self):
        p_ref = self.params.p_ref
        T_prod = self.params.T_prod
        S_o = self.params.S_o
        return Expression(("p_ref", "T_prod", "S_o"), p_ref = p_ref, T_prod = T_prod, S_o = S_o)
    
    def init_variational_form_2D(self):
        W = self.W
        V = self.V
        mesh = self.mesh          
        K_x = self.geo.K_x
        K_y = self.geo.K_y
        #kT = self.geo.kT # need to change this to include water
        ko = self.params.ko
        kw = self.params.kw
        kr = self.params.kr
        phi = self.geo.phi
        c_v_o = self.params.c_v_o
        c_v_w = self.params.c_v_w
        rho_r = self.params.rho_r
        c_r = self.params.c_r
        T_inj = self.params.T_inj
        oil_mu = self.params.oil_mu
        oil_rho = self.params.oil_rho
        water_mu = self.params.water_mu
        water_rho = self.params.water_rho
        rel_perm_o = self.params.rel_perm_o
        rel_perm_w = self.params.rel_perm_w
        
        # Initiate functions
        self.u = Function(W)
        self.u_ = Function(W)

        (p, T, S_o) = split(self.u)
        (p_, T_, S_o_) = split(self.u_)

        q, r, s = TestFunctions(W)
        

        
        # Define facet quantities
        n = FacetNormal(mesh)

        # Define difference between cell centers
        x_func = interpolate(Expression("x[0]"), V)
        y_func = interpolate(Expression("x[1]"), V)
        Delta_h = sqrt(jump(x_func)**2 + jump(y_func)**2)

        # harmonic average for permeability and conductivity
        K_x_facet = conditional(gt(avg(K_x), 0.0), K_x('+')*K_x('-') / avg(K_x), 0.0) 
        K_y_facet = conditional(gt(avg(K_y), 0.0), K_y('+')*K_y('-') / avg(K_y), 0.0)
        
        kT = phi*(S_o*ko + (1-S_o)*kw) + (1-phi)*kr
        
        K_facet = (K_x_facet*(abs(n[0]('+'))+abs(n[0]('-')))/2 + K_y_facet*(abs(n[1]('+'))+abs(n[1]('-')))/2) #need form to be symmetric wrt to '+' and '-'
        
        kT_facet = conditional(gt(avg(kT), 0.0), kT('+')*kT('-') / avg(kT), 0.0)        

        ## Solve a coupled problem 
        # conservation of mass equation WATER - "pressure equation"
        a_accum_w = phi*(water_rho(p,T)*(1.0 - S_o) - water_rho(p_,T_)*(1.0 - S_o_))/self.dt*q*dx
        a_flow_w = K_facet*conditional(gt(jump(p), 0.0), rel_perm_w(S_o('+'))*water_rho(p('+'),T('+'))/water_mu(T('+')), rel_perm_w(S_o('-'))*water_rho(p('-'),T('-'))/water_mu(T('-')))*jump(q)*jump(p)/Delta_h*dS
        # conservation of mass equation OIL - "saturation equation"
        a_accum_o = phi*(oil_rho(p,T)*S_o - oil_rho(p_,T_)*S_o_)/self.dt*s*dx
        a_flow_o = K_facet*conditional(gt(jump(p), 0.0), rel_perm_o(S_o('+'))*oil_rho(p('+'),T('+'))/oil_mu(T('+')), rel_perm_o(S_o('-'))*oil_rho(p('-'),T('-'))/oil_mu(T('-')))*jump(s)*jump(p)/Delta_h*dS
        
        ## WEIGHTED SUM - pressure equation
        a_accum_w = c_v_w*a_accum_w + c_v_o*(phi*(oil_rho(p,T)*S_o - oil_rho(p_,T_)*S_o_)/self.dt*q*dx)
        a_flow_w = c_v_w*a_flow_w + c_v_o*(K_facet*conditional(gt(jump(p), 0.0), rel_perm_o(S_o('+'))*oil_rho(p('+'),T('+'))/oil_mu(T('+')), rel_perm_o(S_o('-'))*oil_rho(p('-'),T('-'))/oil_mu(T('-')))*jump(q)*jump(p)/Delta_h*dS)
        
        # conservation of energy equation
        a_Eaccum = phi*c_v_w*(water_rho(p,T)*(1.0 - S_o)*T - water_rho(p_,T_)*(1.0 - S_o_)*T_)/self.dt*r*dx + phi*c_v_o*(oil_rho(p,T)*S_o*T - oil_rho(p_,T_)*S_o_*T_)/self.dt*r*dx + (1-phi)*rho_r*c_r*(T - T_)/self.dt*r*dx 
        a_advec = K_facet*conditional(gt(jump(p), 0.0), T('+')*rel_perm_w(S_o('+'))*water_rho(p('+'),T('+'))/water_mu(T('+')), T('-')*rel_perm_w(S_o('-'))*water_rho(p('-'),T('-'))/water_mu(T('-')))*c_v_w*jump(r)*jump(p)/Delta_h*dS + K_facet*conditional(gt(jump(p), 0.0), T('+')*rel_perm_o(S_o('+'))*oil_rho(p('+'),T('+'))/oil_mu(T('+')), T('-')*rel_perm_o(S_o('-'))*oil_rho(p('-'),T('-'))/oil_mu(T('-')))*c_v_o*jump(r)*jump(p)/Delta_h*dS
        a_diff = kT_facet*jump(T)/Delta_h*jump(r)*dS

        a = a_accum_w + a_flow_w + a_accum_o + a_flow_o + a_Eaccum + a_diff + a_advec
        self.F = a 
        
        # TO DO: add gravity
        # source terms
        for well in self.case.prod_wells:
            [rate, water_rate, oil_rate] = self.case.flow_rate_twophase(p, T, well, S_o = S_o)
            well.update({'rate': rate})
            well.update({'water_rate': water_rate})
            well.update({'oil_rate': oil_rate})
            tmp_o =  well['delta']*oil_rate
            tmp_w = well['delta']*water_rate
            rhow_o = oil_rho(p, T)
            rhow_w = water_rho(p, T)
            #self.F -= rhow_w*tmp_w*q*dx + rhow_o*tmp_o*s*dx
            self.F -= c_v_w*rhow_w*tmp_w*q*dx + rhow_o*tmp_o*s*dx + c_v_o*rhow_o*tmp_o*q*dx # WEIGHTED SUM
            self.F -= rhow_w*tmp_w*c_v_w*T*r*dx + rhow_o*tmp_o*c_v_o*T*r*dx
        for well in self.case.inj_wells:
            rate = self.case.flow_rate(p, T, well, phase = 'water') # only inject water
            well.update({'rate': rate})
            tmp = well['delta']*rate
            rhow = water_rho(p, T_inj)
            #self.F -= rhow*tmp*q*dx
            self.F -= c_v_w*rhow*tmp*q*dx # WEIGHTED SUM
            self.F -= rhow*tmp*c_v_w*T_inj*r*dx
        for heater in self.case.heaters:
            tmp = heater['delta']
            self.F -= tmp*self.params.U*(T_inj-T)*r*dx
        
    def init_variational_form_3D(self):
        W = self.W
        V = self.V
        mesh = self.mesh          
        K_x = self.geo.K_x
        K_y = self.geo.K_y
        K_z = self.geo.K_z
        #kT = self.geo.kT # need to change this to include water
        ko = self.params.ko
        kw = self.params.kw
        kr = self.params.kr
        phi = self.geo.phi
        c_v_o = self.params.c_v_o
        c_v_w = self.params.c_v_w
        rho_r = self.params.rho_r
        c_r = self.params.c_r
        T_inj = self.params.T_inj
        oil_mu = self.params.oil_mu
        oil_rho = self.params.oil_rho
        water_mu = self.params.water_mu
        water_rho = self.params.water_rho
        rel_perm_o = self.params.rel_perm_o
        rel_perm_w = self.params.rel_perm_w
        g = self.params.g
        
        # Initiate functions
        self.u = Function(W)
        self.u_ = Function(W)

        (p, T, S_o) = split(self.u)
        (p_, T_, S_o_) = split(self.u_)

        q, r, s = TestFunctions(W)
        

        
        # Define facet quantities
        n = FacetNormal(mesh)

        # Define difference between cell centers
        x_func = interpolate(Expression("x[0]"), V)
        y_func = interpolate(Expression("x[1]"), V)
        z_func = interpolate(Expression("x[2]"), V)
        Delta_h = sqrt(jump(x_func)**2 + jump(y_func)**2 + jump(z_func)**2)

        # harmonic average for permeability and conductivity
        K_x_facet = conditional(gt(avg(K_x), 0.0), K_x('+')*K_x('-') / avg(K_x), 0.0) 
        K_y_facet = conditional(gt(avg(K_y), 0.0), K_y('+')*K_y('-') / avg(K_y), 0.0)
        K_z_facet = conditional(gt(avg(K_z), 0.0), K_z('+')*K_z('-') / avg(K_z), 0.0)
        
        kT = phi*(S_o*ko + (1-S_o)*kw) + (1-phi)*kr
        
        K_facet = (K_x_facet*(abs(n[0]('+'))+abs(n[0]('-')))/2 + K_y_facet*(abs(n[1]('+'))+abs(n[1]('-')))/2) #need form to be symmetric wrt to '+' and '-'
        
        kT_facet = conditional(gt(avg(kT), 0.0), kT('+')*kT('-') / avg(kT), 0.0)        
        
        z_flow_w = jump(p)/Delta_h - g*avg(water_rho(p,T))
        z_flow_o = jump(p)/Delta_h - g*avg(oil_rho(p,T))


        ## Solve a coupled problem 
        # conservation of mass equation WATER - "pressure equation"
        a_accum_w = phi*(water_rho(p,T)*(1.0 - S_o) - water_rho(p_,T_)*(1.0 - S_o_))/self.dt*q*dx
        a_flow_w = K_facet*conditional(gt(jump(p), 0.0), rel_perm_w(S_o('+'))*water_rho(p('+'),T('+'))/water_mu(T('+')), rel_perm_w(S_o('-'))*water_rho(p('-'),T('-'))/water_mu(T('-')))*jump(q)*jump(p)/Delta_h*dS_v
        a_flow_w_z = K_z_facet*conditional(gt(z_flow_w, 0.0), rel_perm_w(S_o('+'))*water_rho(p('+'),T('+'))/water_mu(T('+')), rel_perm_w(S_o('-'))*water_rho(p('-'),T('-'))/water_mu(T('-')))*jump(q)*z_flow_w*dS_h
        # conservation of mass equation OIL - "saturation equation"
        a_accum_o = phi*(oil_rho(p,T)*S_o - oil_rho(p_,T_)*S_o_)/self.dt*s*dx
        a_flow_o = K_facet*conditional(gt(jump(p), 0.0), rel_perm_o(S_o('+'))*oil_rho(p('+'),T('+'))/oil_mu(T('+')), rel_perm_o(S_o('-'))*oil_rho(p('-'),T('-'))/oil_mu(T('-')))*jump(s)*jump(p)/Delta_h*dS_v
        a_flow_o_z = K_z_facet*conditional(gt(z_flow_o, 0.0), rel_perm_o(S_o('+'))*oil_rho(p('+'),T('+'))/oil_mu(T('+')), rel_perm_o(S_o('-'))*oil_rho(p('-'),T('-'))/oil_mu(T('-')))*jump(s)*z_flow_o*dS_h
        
        
        ## WEIGHTED SUM - pressure equation
        a_accum_w = c_v_w*a_accum_w + c_v_o*(phi*(oil_rho(p,T)*S_o - oil_rho(p_,T_)*S_o_)/self.dt*q*dx)
        a_flow_w = c_v_w*a_flow_w + c_v_o*(K_facet*conditional(gt(jump(p), 0.0), rel_perm_o(S_o('+'))*oil_rho(p('+'),T('+'))/oil_mu(T('+')), rel_perm_o(S_o('-'))*oil_rho(p('-'),T('-'))/oil_mu(T('-')))*jump(q)*jump(p)/Delta_h*dS_v)
        a_flow_w_z = c_v_w*a_flow_w_z + c_v_o*K_z_facet*conditional(gt(z_flow_o, 0.0), rel_perm_o(S_o('+'))*oil_rho(p('+'),T('+'))/oil_mu(T('+')), rel_perm_o(S_o('-'))*oil_rho(p('-'),T('-'))/oil_mu(T('-')))*jump(q)*z_flow_o*dS_h
        
        # conservation of energy equation
        a_Eaccum = phi*c_v_w*(water_rho(p,T)*(1.0 - S_o)*T - water_rho(p_,T_)*(1.0 - S_o_)*T_)/self.dt*r*dx + phi*c_v_o*(oil_rho(p,T)*S_o*T - oil_rho(p_,T_)*S_o_*T_)/self.dt*r*dx + (1-phi)*rho_r*c_r*(T - T_)/self.dt*r*dx 
        a_advec = K_facet*conditional(gt(jump(p), 0.0), T('+')*rel_perm_w(S_o('+'))*water_rho(p('+'),T('+'))/water_mu(T('+')), T('-')*rel_perm_w(S_o('-'))*water_rho(p('-'),T('-'))/water_mu(T('-')))*c_v_w*jump(r)*jump(p)/Delta_h*dS_v + K_facet*conditional(gt(jump(p), 0.0), T('+')*rel_perm_o(S_o('+'))*oil_rho(p('+'),T('+'))/oil_mu(T('+')), T('-')*rel_perm_o(S_o('-'))*oil_rho(p('-'),T('-'))/oil_mu(T('-')))*c_v_o*jump(r)*jump(p)/Delta_h*dS_v
        a_advec_z = K_z_facet*conditional(gt(z_flow_w, 0.0), T('+')*rel_perm_w(S_o('+'))*water_rho(p('+'),T('+'))/water_mu(T('+')), T('-')*rel_perm_w(S_o('-'))*water_rho(p('-'),T('-'))/water_mu(T('-')))*c_v_w*jump(r)*z_flow_w*dS_h + K_z_facet*conditional(gt(z_flow_o, 0.0), T('+')*rel_perm_o(S_o('+'))*oil_rho(p('+'),T('+'))/oil_mu(T('+')), T('-')*rel_perm_o(S_o('-'))*oil_rho(p('-'),T('-'))/oil_mu(T('-')))*c_v_o*jump(r)*z_flow_o*dS_h
        a_diff = kT_facet*jump(T)/Delta_h*jump(r)*(dS_v + dS_h)

        a = a_accum_w + a_flow_w + a_flow_w_z + a_accum_o + a_flow_o + a_flow_o_z + a_Eaccum + a_advec + a_advec_z + a_diff
        self.F = a 
        
        # source terms
        for well in self.case.prod_wells:
            [rate, water_rate, oil_rate] = self.case.flow_rate_twophase(p, T, well, S_o = S_o)
            well.update({'rate': rate})
            well.update({'water_rate': water_rate})
            well.update({'oil_rate': oil_rate})
            tmp_o =  well['delta']*oil_rate
            tmp_w = well['delta']*water_rate
            rhow_o = oil_rho(p, T)
            rhow_w = water_rho(p, T)
            #self.F -= rhow_w*tmp_w*q*dx + rhow_o*tmp_o*s*dx
            self.F -= c_v_w*rhow_w*tmp_w*q*dx + rhow_o*tmp_o*s*dx + c_v_o*rhow_o*tmp_o*q*dx # WEIGHTED SUM
            self.F -= rhow_w*tmp_w*c_v_w*T*r*dx + rhow_o*tmp_o*c_v_o*T*r*dx
        for well in self.case.inj_wells:
            rate = self.case.flow_rate(p, T, well, phase = 'water') # only inject water
            well.update({'rate': rate})
            tmp = well['delta']*rate
            rhow = water_rho(p, T_inj)
            #self.F -= rhow*tmp*q*dx
            self.F -= c_v_w*rhow*tmp*q*dx # WEIGHTED SUM
            self.F -= rhow*tmp*c_v_w*T_inj*r*dx
        for heater in self.case.heaters:
            tmp = heater['delta']
            self.F -= tmp*self.params.U*(T_inj-T)*r*dx
        
    def init_solver_parameters(self):
        # set to 1 for p,T order, 0 for T,p order in the fieldsplit solver
        # Schur complement always bottom right
        #p_first = 1
        # assign index for subspace
        #if p_first:
            #f0 = 0
            #f1 = 1
            #self.idorder = "pressure-temperature ordering for fieldsplit"
        #else:
            #f0 = 1
            #f1 = 0
            #self.idorder = "temperature-pressure ordering for fieldsplit"
            
        parameters = {
              "snes_type": "newtonls",
              "snes_monitor": True,
              "snes_converged_reason": True, 
              "snes_max_it": 15,
              #"snes_rtol": 1e-10,
              #"snes_view": True,
              
              "ksp_type": "fgmres",
              "ksp_gmres_restart": 40,
              "ksp_max_it": 400,
              #"ksp_rtol": 1e-3,
              #"ksp_view": True,
              #"ksp_monitor_true_residual": True,
              #"ksp_monitor": True,
              "ksp_converged_reason": True, 
              
              #"ksp_pc_side": "right",
              }
        
        
        pc_ilu = {"pc_type": "ilu",
                  "pc_factor_levels": 0,
                  "mat_type": "aij",
                  }
        
        pc_amg = {"pc_type": "hypre",
                  "pc_hypre_type": "boomeramg",
                  "mat_type": "aij",
                  }
        
        pc_gamg = {"pc_type": "gamg",
                   "mat_type": "aij",
                   "mat_block_size": 2,
                   }
        
        pc_lu = {"ksp_type": "preonly",
                 "mat_type": "aij",
                 "pc_type": "lu",
                 }
        
        #block jacobi ilu
        pc_bilu = {"pc_type": "bjacobi",
                   "pc_bjacobi_blocks": 2,
                   "sub_pc_type": "ilu",
                   "sub_pc_factor_levels": 1,
                   "mat_type": "aij",
                   }
        
        richardson_params = {"ksp_type": "richardson",
                             "ksp_max_it": 1,
                             "pc_type": "lu",
                             }
        v_cycle = {"ksp_type": "preonly",
                    "pc_type": "hypre",
                    "pc_hypre_type" : "boomeramg",
                    "pc_hypre_boomeramg_max_iter": 1,
                   }   
        #temperature_params = 
            
        
        pc_cptr = {"pc_type": "composite",
                "pc_composite_type": "multiplicative",
                #"pc_composite_pcs": "python,ilu",
                "pc_composite_pcs": "python,bjacobi",

                "sub_0_pc_python_type": "thermalporous.preconditioners.CPTRStage1_originalPC",
                #"sub_0_cpr_stage1_pc_type": "lu",
                "sub_0_cpr_stage1_pc_type": "fieldsplit",
                #"sub_0_cpr_stage1_pc_fieldsplit_0_fields": 0,
                #"sub_0_cpr_stage1_pc_fieldsplit_1_fields": 1,
                "sub_0_cpr_stage1_pc_fieldsplit_type": "schur",
                "sub_0_cpr_stage1_pc_fieldsplit_schur_fact_type": "FULL",
                
                #"sub_0_cpr_stage1_pc_fieldsplit_schur_precondition": "a11",
                #"sub_0_cpr_stage1_fieldsplit_1": v_cycle,
                
                "sub_0_cpr_stage1_fieldsplit_1_ksp_type": "preonly",
                "sub_0_cpr_stage1_fieldsplit_1_pc_type": "python",
                "sub_0_cpr_stage1_fieldsplit_1_pc_python_type": "thermalporous.preconditioners.ConvDiffSchurTwoPhasesPC",
                "sub_0_cpr_stage1_fieldsplit_1_schur": v_cycle,

                "sub_0_cpr_stage1_fieldsplit_0": v_cycle,
                
                "sub_1_pc_bjacobi_blocks": 1,
                "sub_1_sub_pc_type": "ilu",
                "sub_1_sub_pc_factor_levels": 0,
                "mat_type": "aij",
                }
        
        
        pc_cpr = {"pc_type": "composite",
              "pc_composite_type": "multiplicative",
              #"pc_composite_pcs": "python,ilu",
              "pc_composite_pcs": "python,bjacobi",

              "sub_0_pc_python_type": "thermalporous.preconditioners.CPRStage1PC",
              "sub_0_cpr_stage1": v_cycle,
              
              #"sub_1_pc_type": "ilu",
              #"sub_1_pc_factor_levels": 1,
              
              "sub_1_pc_bjacobi_blocks": 1,
              "sub_1_sub_pc_type": "ilu",
              "sub_1_sub_pc_factor_levels": 0,
              "mat_type": "aij",
              }
        #parameters.update(pc_cptr)
        parameters.update(pc_cpr)
        #parameters.update(pc_ilu)
        #parameters.update(pc_amg)
        #parameters.update(pc_gamg)
        #parameters.update(pc_lu)
        #parameters.update(pc_bilu)
        if self.solver_parameters is None:
            self.solver_parameters = parameters


    #def create_cpr_stage_1(self, snes): 
        #from petsc4py import PETSc
        ## Set fieldsplit indices
        #fdofs = self.W.dof_dset.field_ises  # TODO: should this be local_ises?
        #s_is = PETSc.IS().createGeneral(np.concatenate([iset.indices for iset in fdofs[1:]]))
        #p_is = fdofs[0]
        #cpr_stage1 = CPRStage1(snes, s_is, p_is, True)
        #return cpr_stage1


    def create_cpr_stage_1(self, snes):
        from petsc4py import PETSc
        # Set fieldsplit indices
        fdofs = self.W.dof_dset.field_ises  # TODO: should this be local_ises?
        s_is = fdofs[-1]
        pt_is = PETSc.IS().createGeneral(np.concatenate([iset.indices for iset in fdofs[:-1]]))
        p_is = fdofs[0]
        t_is = fdofs[1]
        cpr_stage1 = CPTRStage1(snes, s_is, pt_is, p_is, t_is)
        return cpr_stage1
 
    @cached_property
    def appctx(self):
        return {"pressure_space": 0, "temperature_space": 1, "saturation_space": 2, "params": self.params, "geo": self.geo, "dt": self.dt, "prod_wells": self.case.prod_wells, "inj_wells": self.case.inj_wells, "case": self.case, "u_": self.u_}
