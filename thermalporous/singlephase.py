from firedrake import *
from thermalporous.thermalmodel import ThermalModel
from thermalporous.preconditioners import ConvDiffSchurPC, CPRStage1PC

from firedrake.utils import cached_property
class SinglePhase(ThermalModel):
    def __init__(self, geo, case, params, end = 1.0, maxdt = 0.005, save = False, n_save = 2, small_dt_start = True, checkpointing = {}, solver_parameters = None, filename = "results/results.txt", dt_init_fact = 2**(-10), vector = False):
        self.name = "Single phase"
        self.geo = geo
        self.case = case
        self.params = params
        self.mesh = geo.mesh
        self.comm = self.mesh.comm
        self.V = geo.V
        if vector:
            self.W = VectorFunctionSpace(self.mesh, "DQ", degree = 0, dim = 2)
        else:
            self.W = self.V*self.V
        self.save = save
        self.n_save = n_save
        self.small_dt_start = small_dt_start
        self.vector = vector
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
        return Constant((p_ref, T_prod))
    
    def init_variational_form_2D(self):
        W = self.W
        V = self.V
        mesh = self.mesh          
        K_x = self.geo.K_x
        K_y = self.geo.K_y
        kT = self.geo.kT
        phi = self.geo.phi
        c_v = self.params.c_v_o
        rho_r = self.params.rho_r
        c_r = self.params.c_r
        T_inj = self.params.T_inj
        oil_mu = self.params.oil_mu
        oil_rho = self.params.oil_rho
        
        # Initiate functions
        self.u = Function(W)
        self.u_ = Function(W)

        (p, T) = split(self.u)
        (p_, T_) = split(self.u_)

        q, r = TestFunctions(W)
        

        
        # Define facet quantities
        n = FacetNormal(mesh)

        # Define difference between cell centers
        x = SpatialCoordinate(V.mesh())
        x_func = interpolate(x[0], V)
        y_func = interpolate(x[1], V)
        Delta_h = sqrt(jump(x_func)**2 + jump(y_func)**2)

        # harmonic average for permeability and conductivity
        K_x_facet = conditional(gt(avg(K_x), 0.0), K_x('+')*K_x('-') / avg(K_x), 0.0) 
        K_y_facet = conditional(gt(avg(K_y), 0.0), K_y('+')*K_y('-') / avg(K_y), 0.0) 
        
        K_facet = (K_x_facet*(abs(n[0]('+'))+abs(n[0]('-')))/2 + K_y_facet*(abs(n[1]('+'))+abs(n[1]('-')))/2) #need form to be symmetric wrt to '+' and '-'
        
        kT_facet = conditional(gt(avg(kT), 0.0), kT('+')*kT('-') / avg(kT), 0.0)        

        # Weight for mass equation
        if False:
            m_w = c_v*self.params.T_prod
        else:
            m_w = 1.0

        ## Solve a coupled problem 
        # conservation of mass equation
        a_accum = phi*(oil_rho(p,T) - oil_rho(p_,T_))/self.dt*q*dx
        a_flow = K_facet*conditional(gt(jump(p), 0.0), oil_rho(p('+'),T('+'))/oil_mu(T('+')), oil_rho(p('-'),T('-'))/oil_mu(T('-')))*jump(q)*jump(p)/Delta_h*dS
        # conservation of energy equation
        a_Eaccum = phi*c_v*(oil_rho(p,T)*T - oil_rho(p_,T_)*T_)/self.dt*r*dx + (1-phi)*rho_r*c_r*(T - T_)/self.dt*r*dx 
        a_advec = K_facet*conditional(gt(jump(p), 0.0), T('+')*oil_rho(p('+'),T('+'))/oil_mu(T('+')), T('-')*oil_rho(p('-'),T('-'))/oil_mu(T('-')))*c_v*jump(r)*jump(p)/Delta_h*dS
        a_diff = kT_facet*jump(T)/Delta_h*jump(r)*dS

        a = m_w*a_accum + m_w*a_flow + a_Eaccum + a_diff + a_advec
        self.F = a

        rhow_o = oil_rho(p, T)
        rhow = oil_rho(p, T_inj)

        ## Source terms using global deltas
        if self.case.name.startswith("Sources"):
            # production wells
            prod_rate = self.case.flow_rate_prod(p, T)
            self.prod_rate = prod_rate
            tmp = self.case.deltas_prod*prod_rate
            self.F -= m_w*rhow_o*tmp*q*dx
            self.F -= rhow_o*tmp*c_v*T*r*dx
            # injection wells
            inj_rate = self.case.flow_rate_inj(p, T)
            self.inj_rate = inj_rate
            tmp = self.case.deltas_inj*inj_rate
            self.F -= m_w*rhow*tmp*q*dx
            self.F -= rhow*tmp*c_v*T_inj*r*dx
            # heaters
            self.F -= self.case.deltas_heaters*self.params.U*(T_inj-T)*r*dx

        ## Source terms using sum of local deltas
        for well in self.case.prod_wells:
            rate = self.case.flow_rate(p, T, well)
            well.update({'rate': rate})
            tmp =  well['delta']*rate
            self.F -= m_w*rhow_o*tmp*q*dx
            self.F -= rhow_o*tmp*c_v*T*r*dx
        for well in self.case.inj_wells:
            rate = self.case.flow_rate(p, T, well)
            well.update({'rate': rate})
            tmp = well['delta']*rate
            self.F -= m_w*rhow*tmp*q*dx
            self.F -= rhow*tmp*c_v*T_inj*r*dx
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
        kT = self.geo.kT
        phi = self.geo.phi
        c_v = self.params.c_v_o
        rho_r = self.params.rho_r
        c_r = self.params.c_r
        T_inj = self.params.T_inj
        oil_mu = self.params.oil_mu
        oil_rho = self.params.oil_rho
        g = self.params.g
        
        # Initiate functions
        self.u = Function(W)
        self.u_ = Function(W)

        (p, T) = split(self.u)
        (p_, T_) = split(self.u_)

        q, r = TestFunctions(W)
        

        
        # Define facet quantities
        n = FacetNormal(mesh)

        # Define difference between cell centers
        x = SpatialCoordinate(V.mesh())
        x_func = interpolate(x[0], V)
        y_func = interpolate(x[1], V)
        z_func = interpolate(x[2], V)
        Delta_h = sqrt(jump(x_func)**2 + jump(y_func)**2 + jump(z_func)**2)

        # harmonic average for permeability and conductivity
        K_x_facet = conditional(gt(avg(K_x), 0.0), K_x('+')*K_x('-') / avg(K_x), 0.0) 
        K_y_facet = conditional(gt(avg(K_y), 0.0), K_y('+')*K_y('-') / avg(K_y), 0.0)
        K_z_facet = conditional(gt(avg(K_z), 0.0), K_z('+')*K_z('-') / avg(K_z), 0.0)
        
        K_facet = (K_x_facet*(abs(n[0]('+'))+abs(n[0]('-')))/2 + K_y_facet*(abs(n[1]('+'))+abs(n[1]('-')))/2) #need form to be symmetric wrt to '+' and '-'
        
        kT_facet = conditional(gt(avg(kT), 0.0), kT('+')*kT('-') / avg(kT), 0.0)        
        
        z_flow = jump(p)/Delta_h - g*avg(oil_rho(p,T))
        
        ## Solve a coupled problem 
        # conservation of mass equation
        a_accum = phi*(oil_rho(p,T) - oil_rho(p_,T_))/self.dt*q*dx
        a_flow = K_facet*conditional(gt(jump(p), 0.0), oil_rho(p('+'),T('+'))/oil_mu(T('+')), oil_rho(p('-'),T('-'))/oil_mu(T('-')))*jump(q)*jump(p)/Delta_h*dS_v
        a_flow_z = K_z_facet*conditional(gt(z_flow, 0.0), oil_rho(p('+'),T('+'))/oil_mu(T('+')), oil_rho(p('-'),T('-'))/oil_mu(T('-')))*jump(q)*z_flow*dS_h
        # conservation of energy equation
        a_Eaccum = phi*c_v*(oil_rho(p,T)*T - oil_rho(p_,T_)*T_)/self.dt*r*dx + (1-phi)*rho_r*c_r*(T - T_)/self.dt*r*dx 
        a_advec = K_facet*conditional(gt(jump(p), 0.0), T('+')*oil_rho(p('+'),T('+'))/oil_mu(T('+')), T('-')*oil_rho(p('-'),T('-'))/oil_mu(T('-')))*c_v*jump(r)*jump(p)/Delta_h*dS_v
        a_advec_z = K_z_facet*conditional(gt(z_flow, 0.0), T('+')*oil_rho(p('+'),T('+'))/oil_mu(T('+')), T('-')*oil_rho(p('-'),T('-'))/oil_mu(T('-')))*c_v*jump(r)*z_flow*dS_h
        a_diff = kT_facet*jump(T)/Delta_h*jump(r)*(dS_v + dS_h)

        a = a_accum + a_flow + a_Eaccum + a_diff + a_advec + a_flow_z + a_advec_z
        self.F = a 

        rhow_o = oil_rho(p, T)
        rhow = oil_rho(p, T_inj)

        ## Source terms using global deltas
        if self.case.name.startswith("Sources"):
            # production wells
            prod_rate = self.case.flow_rate_prod(p, T)
            self.prod_rate = prod_rate
            tmp = self.case.deltas_prod*prod_rate
            self.F -= rhow_o*tmp*q*dx
            self.F -= rhow_o*tmp*c_v*T*r*dx
            # injection wells
            inj_rate = self.case.flow_rate_inj(p, T)
            self.inj_rate = inj_rate
            tmp = self.case.deltas_inj*inj_rate
            self.F -= rhow*tmp*q*dx
            self.F -= rhow*tmp*c_v*T_inj*r*dx
            # heaters
            self.F -= self.case.deltas_heaters*self.params.U*(T_inj-T)*r*dx

        ## Source terms using sum of local deltas
        for well in self.case.prod_wells:
            rate = self.case.flow_rate(p, T, well)
            well.update({'rate': rate})
            tmp =  well['delta']*rate
            self.F -= rhow_o*tmp*q*dx
            self.F -= rhow_o*tmp*c_v*T*r*dx
        for well in self.case.inj_wells:
            rate = self.case.flow_rate(p, T, well)
            well.update({'rate': rate})
            tmp = well['delta']*rate
            self.F -= rhow*tmp*q*dx
            self.F -= rhow*tmp*c_v*T_inj*r*dx
        for heater in self.case.heaters:
            tmp = heater['delta']
            self.F -= tmp*self.params.U*(T_inj-T)*r*dx

    def init_solver_parameters(self):
        # set to 1 for p,T order, 0 for T,p order in the fieldsplit solver
        # Schur complement always bottom right
        p_first = 1
        # assign index for subspace
        if p_first:
            f0 = 0
            f1 = 1
            self.idorder = "pressure-temperature ordering for fieldsplit"
        else:
            f0 = 1
            f1 = 0
            self.idorder = "temperature-pressure ordering for fieldsplit"
            
        parameters = {
                "snes_type": "newtonls",
                "snes_monitor": None,
                "snes_converged_reason": None, 
                "snes_max_it": 15,

                "ksp_type": "fgmres",
                "ksp_converged_reason": None, 

                "ksp_max_it": 200,
                "ksp_gmres_restart": 200,
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

                "sub_1_sub_pc_type": "ilu",
                "sub_1_sub_pc_factor_levels": 0,
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

                "sub_1_sub_pc_type": "ilu",
                "sub_1_sub_pc_factor_levels": 0,
                "mat_type": "aij",
                }
        
        pc_fieldsplit_diag = {"pc_type": "fieldsplit",
                              "pc_fieldsplit_type": "additive",
                              "fieldsplit_0": v_cycle,
                              "fieldsplit_1": v_cycle,
                              }


        pc_hypre = {"pc_type": "hypre",
                "pc_hypre_type": "boomeramg",
                "mat_type": "aij",
                }
     
        pc_ml = {"pc_type": "ml",
                "mat_type": "aij"
                }

        pc_ilu = {"pc_type": "ilu",
                  "pc_factor_levels": 1,
                  "mat_type": "aij",
                  }
        
        pc_amg = {"pc_type": "hypre",
                  "pc_hypre_type": "boomeramg",
                  "mat_type": "aij",
                  }
        
        pc_lu = {"ksp_type": "preonly",
                 "mat_type": "aij",
                 "pc_type": "lu",
                 }
        
        pc_bilu = {"pc_type": "bjacobi",
                   "sub_pc_type": "ilu",
                   "sub_pc_factor_levels": 1,
                   "mat_type": "aij",
                   }
        
        if self.solver_parameters is None:
            self.solver_parameters = "pc_fieldsplit"
        
        if isinstance(self.solver_parameters, str):
            if self.solver_parameters == "pc_fieldsplit_cd":
                parameters.update(pc_fieldsplit_cd)
            elif self.solver_parameters == "pc_fieldsplit_a11":
                parameters.update(pc_fieldsplit_a11)    
            elif self.solver_parameters == "pc_fieldsplit_selfp":
                parameters.update(pc_fieldsplit_selfp)
            elif self.solver_parameters == "pc_cpr":
                parameters.update(pc_cpr)
            elif self.solver_parameters == "pc_cpr_gmres":
                parameters.update(pc_cpr_gmres)
            elif self.solver_parameters == "pc_fieldsplit_diag":
                parameters.update(pc_fieldsplit_diag)
            elif self.solver_parameters == "pc_ilu":
                parameters.update(pc_ilu)
            elif self.solver_parameters == "pc_hypre":    
                parameters.update(pc_hypre)
            elif self.solver_parameters == "pc_lu":    
                parameters.update(pc_lu)
            elif self.solver_parameters == "pc_bilu":    
                parameters.update(pc_bilu)
            self.solver_parameters = parameters

 
    @cached_property
    def appctx(self):
        return {"pressure_space": 0, "temperature_space": 1, "params": self.params, "geo": self.geo, "dt": self.dt, "u_": self.u_, "case": self.case}
