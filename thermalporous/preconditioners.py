import numpy as np

from firedrake import *

from firedrake_citations import Citations
from firedrake.petsc import PETSc

class ConvDiffSchurPC(PCBase):
    def initialize(self, pc):
        from firedrake import TrialFunction, TestFunction, dx, assemble, inner, parameters
        from firedrake.assemble import allocate_matrix, create_assembly_callable
        prefix = pc.getOptionsPrefix()

        # we assume P has things stuffed inside of it
        _, P = pc.getOperators()

        appctx = self.get_appctx(pc)


        case = appctx["case"]
        state = appctx["state"]
        preid = appctx["pressure_space"]
        temid = appctx["temperature_space"]
        params = appctx["params"]
        geo = appctx["geo"]
        V = geo.V
        p0 = split(state)[preid]
        T0 = split(state)[temid]

        T = TrialFunction(V)
        r = TestFunction(V)
        
        dt = appctx["dt"]
        K_x = geo.K_x
        K_y = geo.K_y
        kT = geo.kT
        phi = geo.phi
        c_v = params.c_v_o
        rho_r = params.rho_r
        c_r = params.c_r
        oil_mu = params.oil_mu
        oil_rho = params.oil_rho
        
        # Define facet quantities
        n = FacetNormal(V.mesh())

        # Define difference between cell centers
        x = SpatialCoordinate(V.mesh())
        x_func = interpolate(x[0], V)
        y_func = interpolate(x[1], V)

        # harmonic average for permeability and conductivity
        K_x_facet = conditional(gt(avg(K_x), 0.0), K_x('+')*K_x('-') / avg(K_x), 0.0) 
        K_y_facet = conditional(gt(avg(K_y), 0.0), K_y('+')*K_y('-') / avg(K_y), 0.0) 
        
        K_facet = (K_x_facet*(abs(n[0]('+'))+abs(n[0]('-')))/2 + K_y_facet*(abs(n[1]('+'))+abs(n[1]('-')))/2)
        
        kT_facet = conditional(gt(avg(kT), 0.0), kT('+')*kT('-') / avg(kT), 0.0)  
        
        if geo.dim == 2:
            Delta_h = sqrt(jump(x_func)**2 + jump(y_func)**2)
            # conservation of energy equation
            a_Eaccum = phi*c_v*(oil_rho(p0,T0)*T)/dt*r*dx + (1-phi)*rho_r*c_r*(T)/dt*r*dx 
            a_advec = K_facet*conditional(gt(jump(p0), 0.0), T('+')*oil_rho(p0('+'),T0('+'))/oil_mu(T0('+')), T('-')*oil_rho(p0('-'),T0('-'))/oil_mu(T0('-')))*c_v*jump(r)*jump(p0)/Delta_h*dS
            a_diff = kT_facet*jump(T)/Delta_h*jump(r)*dS
            a = a_Eaccum + a_diff + a_advec
        
        
        if geo.dim == 3:
            K_z = geo.K_z
            g = params.g
            z_func = interpolate(x[2], V)
            Delta_h = sqrt(jump(x_func)**2 + jump(y_func)**2 + jump(z_func)**2)
            K_z_facet = conditional(gt(avg(K_z), 0.0), K_z('+')*K_z('-') / avg(K_z), 0.0)
            z_flow = jump(p0)/Delta_h - g*avg(oil_rho(p0,T0))
            a_Eaccum = phi*c_v*(oil_rho(p0,T0)*T)/dt*r*dx + (1-phi)*rho_r*c_r*(T)/dt*r*dx 
            a_advec = K_facet*conditional(gt(jump(p0), 0.0), T('+')*oil_rho(p0('+'),T0('+'))/oil_mu(T0('+')), T('-')*oil_rho(p0('-'),T0('-'))/oil_mu(T0('-')))*c_v*jump(r)*jump(p0)/Delta_h*dS_v
            a_advec_z = K_z_facet*conditional(gt(z_flow, 0.0), T('+')*oil_rho(p0('+'),T0('+'))/oil_mu(T0('+')), T('-')*oil_rho(p0('-'),T0('-'))/oil_mu(T0('-')))*c_v*jump(r)*z_flow*dS_h
            a_diff = kT_facet*jump(T)/Delta_h*jump(r)*(dS_h + dS_v)
            a = a_Eaccum + a_diff + a_advec + a_advec_z

        rhow = oil_rho(p0, T0)
        # Source terms using global deltas
        if case.name.startswith("Sources"):
            # production wells
            prod_rate = case.flow_rate_prod(p0, T0)
            prod_rate = prod_rate
            tmp = case.deltas_prod*prod_rate
            a -= rhow*tmp*c_v*T*r*dx
            # heaters
            a -= case.deltas_heaters*params.U*(-T)*r*dx
        # Source terms with local deltas
        for well in case.prod_wells:
            rate = case.flow_rate(p0, T0, well)
            tmp =  well['delta']*rate
            a -= rhow*tmp*c_v*T*r*dx
        for heater in case.heaters:
            tmp = heater['delta']
            a -= tmp*params.U*(-T)*r*dx

        opts = PETSc.Options()
        self.mat_type = opts.getString(prefix+"schur_mat_type", parameters["default_matrix_type"])

        self.A = allocate_matrix(a, form_compiler_parameters=None,
                                  mat_type=self.mat_type)
        self._assemble_A = create_assembly_callable(a, tensor=self.A,
                                                     form_compiler_parameters=None,
                                                     mat_type=self.mat_type)
        self._assemble_A()
        self.A.force_evaluation()

        Pmat = self.A.petscmat
        Pmat.setNullSpace(P.getNullSpace())
        tnullsp = P.getTransposeNullSpace()
        if tnullsp.handle != 0:
            Pmat.setTransposeNullSpace(tnullsp)

        ksp = PETSc.KSP().create()
        ksp.incrementTabLevel(1, parent=pc)
        ksp.setOperators(Pmat)
        ksp.setOptionsPrefix(prefix + "schur_")
        ksp.setUp()
        ksp.setFromOptions()
        self.ksp = ksp


    def update(self, pc):
        self._assemble_A()
        self.A.force_evaluation()


    def apply(self, pc, X, Y):
        self.ksp.solve(X, Y)


    # should not be used
    applyTranspose = apply

    def view(self, pc, viewer=None):
        super(ConvDiffSchurPC, self).view(pc, viewer)
        viewer.printfASCII("KSP solver for S^-1\n")
        self.ksp.view(viewer)    
        
class ConvDiffSchurTwoPhasesPC(PCBase):
    def initialize(self, pc):
        from firedrake import TrialFunction, TestFunction, dx, assemble, inner, parameters
        from firedrake.assemble import allocate_matrix, create_assembly_callable

        prefix = pc.getOptionsPrefix()

        # we assume P has things stuffed inside of it
        _, P = pc.getOperators()
        appctx = self.get_appctx(pc)

        case = appctx["case"]
        state = appctx["state"]
        preid = appctx["pressure_space"]
        temid = appctx["temperature_space"]
        Satid = appctx["saturation_space"]
        params = appctx["params"]
        geo = appctx["geo"]
        V = geo.V
        dim = geo.dim
        p0 = split(state)[preid]
        T0 = split(state)[temid]
        S0 = split(state)[Satid]

        T = TrialFunction(V)
        r = TestFunction(V)

        dt = appctx["dt"]
        K_x = geo.K_x
        K_y = geo.K_y
        phi = geo.phi
        kT = phi*(S0*params.ko + (1-S0)*params.kw) + (1-phi)*params.kr
        c_v_o = params.c_v_o
        c_v_w = params.c_v_w
        rho_r = params.rho_r
        c_r = params.c_r
        oil_mu = params.oil_mu
        oil_rho = params.oil_rho
        water_mu = params.water_mu
        water_rho = params.water_rho
        rel_perm_o = params.rel_perm_o
        rel_perm_w = params.rel_perm_w
        
        # Define facet quantities
        n = FacetNormal(V.mesh())

        # Define difference between cell centers
        x = SpatialCoordinate(V.mesh())
        x_func = interpolate(x[0], V)
        y_func = interpolate(x[1], V)
        
        # harmonic average for permeability and conductivity
        K_x_facet = conditional(gt(avg(K_x), 0.0), K_x('+')*K_x('-') / avg(K_x), 0.0) 
        K_y_facet = conditional(gt(avg(K_y), 0.0), K_y('+')*K_y('-') / avg(K_y), 0.0) 
        
        K_facet = (K_x_facet*(abs(n[0]('+'))+abs(n[0]('-')))/2 + K_y_facet*(abs(n[1]('+'))+abs(n[1]('-')))/2)
        
        kT_facet = conditional(gt(avg(kT), 0.0), kT('+')*kT('-') / avg(kT), 0.0) 
        
        if geo.dim == 2:
            Delta_h = sqrt(jump(x_func)**2 + jump(y_func)**2)
            # conservation of energy equation
            a_Eaccum = phi*c_v_o*S0*(oil_rho(p0,T0)*T)/dt*r*dx + phi*c_v_w*(1.0 - S0)*(water_rho(p0,T0)*T)/dt*r*dx + (1-phi)*rho_r*c_r*(T)/dt*r*dx 
            a_advec = K_facet*conditional(gt(jump(p0), 0.0), T('+')*rel_perm_w(S0('+'))*water_rho(p0('+'),T0('+'))/water_mu(T0('+')), T('-')*rel_perm_w(S0('-'))*water_rho(p0('-'),T0('-'))/water_mu(T0('-')))*c_v_w*jump(r)*jump(p0)/Delta_h*dS + K_facet*conditional(gt(jump(p0), 0.0), T('+')*rel_perm_o(S0('+'))*oil_rho(p0('+'),T0('+'))/oil_mu(T0('+')), T('-')*rel_perm_o(S0('-'))*oil_rho(p0('-'),T0('-'))/oil_mu(T0('-')))*c_v_o*jump(r)*jump(p0)/Delta_h*dS
            a_diff = kT_facet*jump(T)/Delta_h*jump(r)*dS
            
            a = a_Eaccum + a_diff + a_advec
        
        if geo.dim == 3:
            K_z = geo.K_z
            g = params.g
            z_func = interpolate(x[2], V)
            Delta_h = sqrt(jump(x_func)**2 + jump(y_func)**2 + jump(z_func)**2)
            K_z_facet = conditional(gt(avg(K_z), 0.0), K_z('+')*K_z('-') / avg(K_z), 0.0)
            z_flow_w = jump(p0)/Delta_h - g*avg(water_rho(p0,T0))
            z_flow_o = jump(p0)/Delta_h - g*avg(oil_rho(p0,T0))
            
            a_Eaccum = phi*c_v_o*S0*(oil_rho(p0,T0)*T)/dt*r*dx + phi*c_v_w*(1.0 - S0)*(water_rho(p0,T0)*T)/dt*r*dx + (1-phi)*rho_r*c_r*(T)/dt*r*dx 
            a_advec = K_facet*conditional(gt(jump(p0), 0.0), T('+')*rel_perm_w(S0('+'))*water_rho(p0('+'),T0('+'))/water_mu(T0('+')), T('-')*rel_perm_w(S0('-'))*water_rho(p0('-'),T0('-'))/water_mu(T0('-')))*c_v_w*jump(r)*jump(p0)/Delta_h*dS_v + K_facet*conditional(gt(jump(p0), 0.0), T('+')*rel_perm_o(S0('+'))*oil_rho(p0('+'),T0('+'))/oil_mu(T0('+')), T('-')*rel_perm_o(S0('-'))*oil_rho(p0('-'),T0('-'))/oil_mu(T0('-')))*c_v_o*jump(r)*jump(p0)/Delta_h*dS_v
            a_advec_z = K_z_facet*conditional(gt(z_flow_w, 0.0), T('+')*rel_perm_w(S0('+'))*water_rho(p0('+'),T0('+'))/water_mu(T0('+')), T('-')*rel_perm_w(S0('-'))*water_rho(p0('-'),T0('-'))/water_mu(T0('-')))*c_v_w*jump(r)*z_flow_w*dS_h + K_z_facet*conditional(gt(z_flow_o, 0.0), T('+')*rel_perm_o(S0('+'))*oil_rho(p0('+'),T0('+'))/oil_mu(T0('+')), T('-')*rel_perm_o(S0('-'))*oil_rho(p0('-'),T0('-'))/oil_mu(T0('-')))*c_v_o*jump(r)*z_flow_o*dS_h
            a_diff = kT_facet*jump(T)/Delta_h*jump(r)*(dS_h + dS_v)
            a = a_Eaccum + a_diff + a_advec + a_advec_z


        rhow_o = oil_rho(p0, T0)
        rhow_w = water_rho(p0, T0)
        # Source terms using global deltas
        if case.name.startswith("Sources"):
            # production wells
            [rate, water_rate, oil_rate] = case.flow_rate_twophase_prod(p0, T0, S_o = S0)
            tmp_o = case.deltas_prod*oil_rate
            tmp_w = case.deltas_prod*water_rate
            a -= rhow_w*tmp_w*c_v_w*T*r*dx + rhow_o*tmp_o*c_v_o*T*r*dx
            # heaters
            a -= case.deltas_heaters*params.U*(-T)*r*dx
        # Source terms with local deltas
        for well in case.prod_wells:
            [rate, water_rate, oil_rate] = case.flow_rate_twophase(p0, T0, well, S_o = S0)
            tmp_o =  well['delta']*oil_rate
            tmp_w = well['delta']*water_rate
            a -= rhow_w*tmp_w*c_v_w*T*r*dx + rhow_o*tmp_o*c_v_o*T*r*dx
        for heater in case.heaters:
            tmp = heater['delta']
            a -= tmp*params.U*(-T)*r*dx

        opts = PETSc.Options()
        self.mat_type = opts.getString(prefix+"schur_mat_type", parameters["default_matrix_type"])
        
        self.A = allocate_matrix(a, form_compiler_parameters=None,
                                  mat_type=self.mat_type)
        self._assemble_A = create_assembly_callable(a, tensor=self.A,
                                                     form_compiler_parameters=None,
                                                     mat_type=self.mat_type)
        self._assemble_A()
        self.A.force_evaluation()

        Pmat = self.A.petscmat
        Pmat.setNullSpace(P.getNullSpace())
        tnullsp = P.getTransposeNullSpace()
        if tnullsp.handle != 0:
            Pmat.setTransposeNullSpace(tnullsp)

        ksp = PETSc.KSP().create()
        ksp.incrementTabLevel(1, parent=pc)
        ksp.setOperators(Pmat)
        ksp.setOptionsPrefix(prefix + "schur_")
        
        ksp.setUp()
        ksp.setFromOptions()
        self.ksp = ksp


    def update(self, pc):
        self._assemble_A()
        self.A.force_evaluation()


    def apply(self, pc, X, Y):
        self.ksp.solve(X, Y)


    # should not be used
    applyTranspose = apply

    def view(self, pc, viewer=None):
        super(ConvDiffSchurTwoPhasesPC, self).view(pc, viewer)
        viewer.printfASCII("KSP solver for S^-1\n")
        self.ksp.view(viewer)    
        
#class ConvDiffSchurTwoPhasesPC(PCBase):
    #def initialize(self, pc):
        #from firedrake import TrialFunction, TestFunction, dx, assemble, inner, parameters
        #from firedrake.assemble import allocate_matrix, create_assembly_callable
        #prefix = pc.getOptionsPrefix()

        ## we assume P has things stuffed inside of it
        #_, P = pc.getOperators()
        ##from IPython import embed; embed()
        #appctx = self.get_appctx(pc)

        #case = appctx["case"]
        #state = appctx["state"]
        #preid = appctx["pressure_space"]
        #temid = appctx["temperature_space"]
        #Satid = appctx["saturation_space"]
        #params = appctx["params"]
        #geo = appctx["geo"]
        #V = geo.V
        #p0 = split(state)[preid]
        #T0 = split(state)[temid]
        #S0 = split(state)[Satid]

        #T = TrialFunction(V)
        #r = TestFunction(V)
        ## Handle vector and tensor-valued spaces.
        
        ## Maybe don't have to reconstruct everything?
        #dt = appctx["dt"]
        #K_x = geo.K_x
        #K_y = geo.K_y
        #kT = geo.kT
        #phi = geo.phi
        #c_v_o = params.c_v_o
        #c_v_w = params.c_v_w
        #rho_r = params.rho_r
        #c_r = params.c_r
        #oil_mu = params.oil_mu
        #oil_rho = params.oil_rho
        #water_mu = params.water_mu
        #water_rho = params.water_rho
        #rel_perm_o = params.rel_perm_o
        #rel_perm_w = params.rel_perm_w
        
        ## Define facet quantities
        #n = FacetNormal(V.mesh())

        ## Define difference between cell centers
        #x = SpatialCoordinate(V.mesh())
        #x_func = interpolate(x[0], V)
        #y_func = interpolate(x[1], V)
        #Delta_h = sqrt(jump(x_func)**2 + jump(y_func)**2)

        ## harmonic average for permeability and conductivity
        #K_x_facet = conditional(gt(avg(K_x), 0.0), K_x('+')*K_x('-') / avg(K_x), 0.0) 
        #K_y_facet = conditional(gt(avg(K_y), 0.0), K_y('+')*K_y('-') / avg(K_y), 0.0) 
        
        #K_facet = (K_x_facet*(abs(n[0]('+'))+abs(n[0]('-')))/2 + K_y_facet*(abs(n[1]('+'))+abs(n[1]('-')))/2)
        
        #kT_facet = conditional(gt(avg(kT), 0.0), kT('+')*kT('-') / avg(kT), 0.0)  
        ## conservation of energy equation
        #a_Eaccum = phi*c_v_o*S0*(oil_rho(p0,T0)*T)/dt*r*dx + phi*c_v_w*(1.0 - S0)*(water_rho(p0,T0)*T)/dt*r*dx + (1-phi)*rho_r*c_r*(T)/dt*r*dx 
        #a_advec = K_facet*conditional(gt(jump(p0), 0.0), T('+')*rel_perm_w(S0('+'))*water_rho(p0('+'),T0('+'))/water_mu(T0('+')), T('-')*rel_perm_w(S0('-'))*water_rho(p0('-'),T0('-'))/water_mu(T0('-')))*c_v_w*jump(r)*jump(p0)/Delta_h*dS + K_facet*conditional(gt(jump(p0), 0.0), T('+')*rel_perm_o(S0('+'))*oil_rho(p0('+'),T0('+'))/oil_mu(T0('+')), T('-')*rel_perm_o(S0('-'))*oil_rho(p0('-'),T0('-'))/oil_mu(T0('-')))*c_v_o*jump(r)*jump(p0)/Delta_h*dS
        #a_diff = kT_facet*jump(T)/Delta_h*jump(r)*dS
        
        #a = a_Eaccum + a_diff + a_advec

        ## source terms
        #for well in case.prod_wells:
            #[rate, water_rate, oil_rate] = case.flow_rate_twophase(p0, T0, well, S_o = S0)
            ##well.update({'rate': rate})
            #tmp_o =  well['delta']*oil_rate
            #tmp_w = well['delta']*water_rate
            #rhow_o = oil_rho(p0, T0)
            #rhow_w = water_rho(p0, T0)
            #a -= rhow_w*tmp_w*c_v_w*T*r*dx + rhow_o*tmp_o*c_v_o*T*r*dx

        #opts = PETSc.Options()
        #self.mat_type = opts.getString(prefix+"schur_mat_type", parameters["default_matrix_type"])
        
        #self.A = allocate_matrix(a, form_compiler_parameters=None,
                                  #mat_type=self.mat_type)
        #self._assemble_A = create_assembly_callable(a, tensor=self.A,
                                                     #form_compiler_parameters=None,
                                                     #mat_type=self.mat_type)
        #self._assemble_A()
        #self.A.force_evaluation()

        #Pmat = self.A.petscmat
        #Pmat.setNullSpace(P.getNullSpace())
        #tnullsp = P.getTransposeNullSpace()
        #if tnullsp.handle != 0:
            #Pmat.setTransposeNullSpace(tnullsp)

        #ksp = PETSc.KSP().create()
        #ksp.incrementTabLevel(1, parent=pc)
        #ksp.setOperators(Pmat)
        #ksp.setOptionsPrefix(prefix + "schur_")
        
        #ksp.setUp()
        #ksp.setFromOptions()
        #self.ksp = ksp


    #def update(self, pc):
        #self._assemble_A()
        #self.A.force_evaluation()


    #def apply(self, pc, X, Y):
        #self.ksp.solve(X, Y)


    ## should not be used
    #applyTranspose = apply

    #def view(self, pc, viewer=None):
        #super(ConvDiffSchurTwoPhasesPC, self).view(pc, viewer)
        #viewer.printfASCII("KSP solver for S^-1\n")
        #self.ksp.view(viewer)   

from petsc4py import PETSc

class CPTRStage1:
    '''
    1st stage solver for constrained pressure-temperature residual
    '''
    def __init__(self, snes, s_is, pt_is, p_is, t_is):
        self.snes = snes

        self.s_is_full, self.pt_is_full = (s_is, pt_is)
        
        self.p_is = p_is
        self.t_is = t_is

    def construct_is(self):
        snes = self.snes
        
        # here p stands for both pressure and temperature
        s_is, pt_is = (self.s_is_full, self.pt_is_full)
        self.s_is, self.pt_is = (s_is, pt_is)
        self.rows, self.cols = ([s_is, s_is, pt_is, pt_is],
                                [s_is, pt_is, s_is, pt_is])

    def setUp(self, pc):
        self.construct_is()
        prefix = pc.getOptionsPrefix()

        A, P = pc.getOperators()

        App = A.createSubMatrix(self.rows[3], self.cols[3])
                              
        Atildepp = App # Weighted Sum is done in the weak form

        pc_schur = PETSc.PC().create()
        #pc_schur.setType("lu")
        #pc_schur.setType("hypre")
        #pc_schur.setType("ksp")
        pc_schur.setOptionsPrefix(prefix + "cpr_stage1_")
        pc_schur.setFromOptions()
        pc_schur.setOperators(Atildepp)
        
        pc_schur.setFieldSplitIS(("0", self.p_is), ("1", self.t_is))
        
        pc_schur.setUp()
        
        self.pc_schur = pc_schur

        
    def apply(self, pc, x, y):
        # x is the input Vec to this preconditioner
        # y is the output Vec, P^{-1} x
        x_s = x.getSubVector(self.s_is)
        x_p = x.getSubVector(self.pt_is)
        y_p = y.getSubVector(self.pt_is)
        r_p = x_p.copy()
        
        self.pc_schur.apply(r_p, y_p)
        
        # need to do something for y_s
        y_s = y.getSubVector(self.s_is)
        y_s.set(0.0)


class CPRStage1PC(PCBase):
    '''
    1st stage solver for constrained pressure residual
    '''
    def initialize(self, pc):
        
        prefix = pc.getOptionsPrefix()
        appctx = self.get_appctx(pc)
        
        V = appctx["geo"].V
        if "saturation_space" in appctx:
            W = V*V*V
        else:
            W = V*V
        
        
        fdofs = W.dof_dset.field_ises 
        self.nonp_is = PETSc.IS().createGeneral(np.concatenate([iset.indices for iset in fdofs[1:]])) # non-pressure fields: saturations and temperatures in the two-phase case
        self.s_is = fdofs[-1] # in singlephase: temperature. in multiphase: saturations
        #self.s_is = self.nonp_is
        self.T_is = fdofs[1] # temperature dofs
        self.p_is = fdofs[0]
        
        self.decoup = "No"
        #self.decoup = "QI"
        #self.decoup = "TI"
        #self.decoup = "QI_temp"
        #self.decoup = "TI_temp"
        
        
        if self.decoup == "TI":
            self.create_blocks = self.create_blocks_TI
        elif self.decoup == "TI_temp":
            self.create_blocks = self.create_blocks_TI_temp
        elif self.decoup == "QI":
            self.create_blocks = self.create_blocks_QI
        elif self.decoup == "QI_temp":
            self.create_blocks = self.create_blocks_QI_temp
        elif self.decoup == "No":
            self.create_blocks = self.create_blocks_
        self.create_blocks(pc)

        pc_schur = PETSc.PC().create()
        #pc_schur.setType("lu")
        #pc_schur.setType("hypre")
        #pc_schur.setType("ksp")
        pc_schur.setOptionsPrefix(prefix + "cpr_stage1_")
        pc_schur.setFromOptions()
        pc_schur.setOperators(self.Atildepp)
        
        pc_schur.setUp()
        
        self.pc_schur = pc_schur
        
        if not self.decoup == "No":
            psize = self.p_is.getSize()
            # Vector for storing preconditioned pressure-vector
            self.workVec = PETSc.Vec().create()
            self.workVec.setSizes(psize)
            self.workVec.setUp()

    def create_blocks_(self, pc):
        A, P = pc.getOperators()
        App = A.createSubMatrix(self.p_is, self.p_is)
        self.Atildepp = App # Weighted Sum is done in the weak form
        
    def create_blocks_TI(self, pc):
        A, P = pc.getOperators()

        App = A.createSubMatrix(self.p_is, self.p_is)
                              
        # Decoupling operators        
        Ass = A.createSubMatrix(self.s_is, self.s_is)
        Asp = A.createSubMatrix(self.s_is, self.p_is)
        Aps = A.createSubMatrix(self.p_is, self.s_is)
        
        #True-IMPES
        AssT = PETSc.Mat()
        Ass.transpose(out=AssT)
        invdiag = AssT.getRowSum()
        invdiag.reciprocal()
        invDss = PETSc.Mat().create()
        invDss.setSizes(Ass.getSizes())
        invDss.setUp()
        invDss.setDiagonal(invdiag)
        
        ApsT = PETSc.Mat()
        Aps.transpose(out=ApsT)
        diag = ApsT.getRowSum()
        Dps = PETSc.Mat().create()
        Dps.setSizes(Aps.getSizes())
        Dps.setUp()
        Dps.setDiagonal(diag)
        
        self.apsinvdss = Dps.matMult(invDss)
        self.Atildepp = App - self.apsinvdss.matMult(Asp) # Weighted Sum is done in the weak form
        

    def create_blocks_TI_temp(self, pc):
        import numpy as np
        A, P = pc.getOperators()

        App = A.createSubMatrix(self.p_is, self.p_is)
                              
        # Decoupling operators        
        Ass = A.createSubMatrix(self.nonp_is, self.nonp_is)
        Asp = A.createSubMatrix(self.nonp_is, self.p_is)
        Aps = A.createSubMatrix(self.p_is, self.nonp_is)
        indices = Aps.getOwnershipRange()
        
        ASS = A.createSubMatrix(self.s_is, self.s_is)
        AST = A.createSubMatrix(self.s_is, self.T_is)
        ATS = A.createSubMatrix(self.T_is, self.s_is)
        ATT = A.createSubMatrix(self.T_is, self.T_is)
        ApS = A.createSubMatrix(self.p_is, self.s_is)
        ApT = A.createSubMatrix(self.p_is, self.T_is)
        
        A_T = PETSc.Mat()
        ASS.transpose(out=A_T)
        Rsum_SS = A_T.getRowSum()
        AST.transpose(out=A_T)
        Rsum_ST = A_T.getRowSum()
        ATS.transpose(out=A_T)
        Rsum_TS = A_T.getRowSum()
        ATT.transpose(out=A_T)
        Rsum_TT = A_T.getRowSum()
        ApT.transpose(out=A_T)
        Rsum_pT = A_T.getRowSum() 
        ApS.transpose(out=A_T)
        Rsum_pS = A_T.getRowSum() 
        
        
        
        #True-IMPES
        invDss = PETSc.Mat().create()
        invDss.setSizes(Ass.getSizes())
        invDss.setUp()
        invDss.assemblyBegin()
        block = np.zeros([2,2])      
        s_size = self.s_is.getSize()
        

        for i in range(indices[0], indices[-1]):
            #print(rank, i, indices)
            block[0,0] = Rsum_TT[i]
            block[0,1] = Rsum_TS[i]
            block[1,0] = Rsum_ST[i]
            block[1,1] = Rsum_SS[i]
            block = np.linalg.inv(block)
            invDss.setValue(i,i, block[0,0])
            invDss.setValue(i,i + s_size, block[0,1])
            invDss.setValue(i + s_size,i, block[1,0])
            invDss.setValue(i + s_size,i + s_size, block[1,1])
        invDss.assemblyEnd()

        Dps = PETSc.Mat().create()
        Dps.setSizes(Aps.getSizes())
        Dps.setUp()
        Dps.assemblyBegin()
        for i in range(indices[0], indices[-1]):
            Dps.setValue(i,i, Rsum_pT[i])
            Dps.setValue(i,i + s_size, Rsum_pS[i])
        Dps.assemblyEnd()

        self.apsinvdss = Dps.matMult(invDss)
        self.Atildepp = App - self.apsinvdss.matMult(Asp) # Weighted Sum is done in the weak form
        
    def create_blocks_QI(self, pc):
        A, P = pc.getOperators()

        App = A.createSubMatrix(self.p_is, self.p_is)
                              
        # Decoupling operators        
        Ass = A.createSubMatrix(self.s_is, self.s_is)
        Asp = A.createSubMatrix(self.s_is, self.p_is)
        Aps = A.createSubMatrix(self.p_is, self.s_is)
        
        #Quasi-IMPES
        invdiag = Ass.getDiagonal()
        invdiag.reciprocal()
        invDss = PETSc.Mat().create()
        invDss.setSizes(Ass.getSizes())
        invDss.setUp()
        invDss.setDiagonal(invdiag)
        
        diag = Aps.getDiagonal()
        Dps = PETSc.Mat().create()
        Dps.setSizes(Aps.getSizes())
        Dps.setUp()
        Dps.setDiagonal(diag)
        self.apsinvdss = Dps.matMult(invDss)
        self.Atildepp = App - self.apsinvdss.matMult(Asp) # Weighted Sum is done in the weak form
        
    def create_blocks_QI_temp(self, pc):
        import numpy as np
        A, P = pc.getOperators()

        App = A.createSubMatrix(self.p_is, self.p_is)
                              
        # Decoupling operators        
        Ass = A.createSubMatrix(self.nonp_is, self.nonp_is)
        Asp = A.createSubMatrix(self.nonp_is, self.p_is)
        Aps = A.createSubMatrix(self.p_is, self.nonp_is)
        indices = Aps.getOwnershipRange()
        
        
        ASS = A.createSubMatrix(self.s_is, self.s_is)
        AST = A.createSubMatrix(self.s_is, self.T_is)
        ATS = A.createSubMatrix(self.T_is, self.s_is)
        ATT = A.createSubMatrix(self.T_is, self.T_is)
        ApS = A.createSubMatrix(self.p_is, self.s_is)
        ApT = A.createSubMatrix(self.p_is, self.T_is)
        
        A_T = PETSc.Mat()
        ASS.transpose(out=A_T)
        Rsum_SS = ASS.getDiagonal()
        Rsum_ST = AST.getDiagonal()
        Rsum_TS = ATS.getDiagonal()
        Rsum_TT = ATT.getDiagonal()
        Rsum_pT = ApT.getDiagonal() 
        Rsum_pS = ApS.getDiagonal() 
        
        
        #Quasi-IMPES
        invDss = PETSc.Mat().create()
        invDss.setSizes(Ass.getSizes())
        invDss.setUp()
        invDss.assemblyBegin()
        block = np.zeros([2,2])      
        s_size = self.s_is.getSize()
        #local_size = self.s_is.getLocalSize()
        #r_is = range(indices[0], indices[-1])
        #from IPython import embed; embed()
        for i in range(indices[0], indices[-1]):
            #print(rank, i, indices)
            block[0,0] = Rsum_TT[i]
            block[0,1] = Rsum_TS[i]
            block[1,0] = Rsum_ST[i]
            block[1,1] = Rsum_SS[i]
            block = np.linalg.inv(block)
            invDss.setValue(i,i, block[0,0])
            invDss.setValue(i,i + s_size, block[0,1])
            invDss.setValue(i + s_size,i, block[1,0])
            invDss.setValue(i + s_size,i + s_size, block[1,1])
        #for i in range(local_size):
            #block[0,0] = Ass.getValue(i, i)
            #block[0,1] = Ass.getValue(i, i + s_size)
            #block[1,0] = Ass.getValue(i + local_size, i)
            #block[1,1] = Ass.getValue(i + local_size, i + s_size)
            #block = np.linalg.inv(block)
            #invDss.setValue(r_is[i], r_is[i], block[0,0])
            #invDss.setValue(r_is[i], r_is[i] + s_size, block[0,1])
            #invDss.setValue(r_is[i] + s_size, r_is[i], block[1,0])
            #invDss.setValue(r_is[i] + s_size, r_is[i] + s_size, block[1,1])
        invDss.assemblyEnd()

        indices = Aps.getOwnershipRange()
        Dps = PETSc.Mat().create()
        Dps.setSizes(Aps.getSizes())
        Dps.setUp()
        Dps.assemblyBegin()
        for i in range(indices[0], indices[-1]):
            Dps.setValue(i,i, Aps.getValue(i,i))
            Dps.setValue(i,i + s_size, Aps.getValue(i,i+s_size))
        Dps.assemblyEnd()

        self.apsinvdss = Dps.matMult(invDss)
        self.Atildepp = App - self.apsinvdss.matMult(Asp) # Weighted Sum is done in the weak form
        
    def update(self, pc):
        self.create_blocks(pc)
        self.pc_schur.setOperators(self.Atildepp)

    
    def apply(self, pc, x, y):
        # x is the input Vec to this preconditioner
        # y is the output Vec, P^{-1} x
        x_p = x.getSubVector(self.p_is)
        y_p = y.getSubVector(self.p_is)
        r_p = x_p.copy()
        
        if not self.decoup == "No":
            if self.decoup == "QI_temp" or self.decoup == "TI_temp":
                x_s = x.getSubVector(self.nonp_is)
            else:
                x_s = x.getSubVector(self.s_is)
            self.apsinvdss.mult(x_s, self.workVec)
            r_p.axpy(  -1.0, self.workVec)  # x_p - Aps*inv(Dss)*x_s
        
        self.pc_schur.apply(r_p, y_p)

        # need to do something for y_s
        y_s = y.getSubVector(self.nonp_is)
        y_s.set(0.0)

    # should not be used
    applyTranspose = apply

class CTRStage1PC(PCBase):
    '''
    1st stage solver for constrained temperature residual
    '''
    def initialize(self, pc):
        
        prefix = pc.getOptionsPrefix()

        A, P = pc.getOperators()
        appctx = self.get_appctx(pc)
        
        V = appctx["geo"].V
        if "saturation_space" in appctx:
            W = V*V*V
        else:
            W = V*V
        
        fdofs = W.dof_dset.field_ises  # TODO: should this be local_ises?
        self.s_is = PETSc.IS().createGeneral(np.concatenate([iset.indices for iset in fdofs[0:2:1]]))
        self.p_is = fdofs[1]
        
        App = A.createSubMatrix(self.p_is, self.p_is)
                              
        self.Atildepp = App # Weighted Sum is done in the weak form

        pc_schur = PETSc.PC().create()
        #pc_schur.setType("lu")
        #pc_schur.setType("hypre")
        #pc_schur.setType("ksp")
        pc_schur.setOptionsPrefix(prefix + "cpr_stage1_")
        pc_schur.setFromOptions()
        pc_schur.setOperators(self.Atildepp)
        
        pc_schur.setUp()
        
        self.pc_schur = pc_schur


    def update(self, pc):
        A, P = pc.getOperators()
        App = A.createSubMatrix(self.p_is, self.p_is)              
        self.Atildepp = App 
        self.pc_schur.setOperators(self.Atildepp)

    
    def apply(self, pc, x, y):
        # x is the input Vec to this preconditioner
        # y is the output Vec, P^{-1} x
        x_s = x.getSubVector(self.s_is)
        x_p = x.getSubVector(self.p_is)
        y_p = y.getSubVector(self.p_is)
        r_p = x_p.copy()
        
        self.pc_schur.apply(r_p, y_p)

        # need to do something for y_s
        y_s = y.getSubVector(self.s_is)
        y_s.set(0.0)

    # should not be used
    applyTranspose = apply

class CPTRStage1PC(PCBase):
    '''
    1st stage solver for constrained pressure-temperature residual
    '''
    def initialize(self, pc):
        from firedrake import TrialFunction, TestFunction, dx, assemble, inner, parameters
        from firedrake.assemble import allocate_matrix, create_assembly_callable
        from firedrake.formmanipulation import ExtractSubBlock
        prefix = pc.getOptionsPrefix()
        #from IPython import embed; embed()
        # we assume P has things stuffed inside of it
        _, P = pc.getOperators()
        
        appctx = self.get_appctx(pc)

        case = appctx["case"]
        u = appctx["state"]
        preid = appctx["pressure_space"]
        temid = appctx["temperature_space"]
        Satid = appctx["saturation_space"]
        params = appctx["params"]
        geo = appctx["geo"]
        
        V = geo.V
        W = V*V*V
        
        fdofs = W.dof_dset.field_ises  # TODO: should this be local_ises?
        self.s_is = fdofs[-1]
        self.pt_is = PETSc.IS().createGeneral(np.concatenate([iset.indices for iset in fdofs[:-1]]))
        self.p_is = fdofs[0]
        self.t_is = fdofs[1]
        
        u_ = appctx["u_"]
        p_ = split(u_)[preid]
        T_ = split(u_)[temid]
        S_o_ = split(u_)[Satid]
        p = split(u)[preid]
        T = split(u)[temid]
        S_o = split(u)[Satid]
        q, r, s = TestFunction(W)
        # Handle vector and tensor-valued spaces.
        
        # Maybe don't have to reconstruct everything?
        dt = appctx["dt"]
        K_x = geo.K_x
        K_y = geo.K_y
        kT = geo.kT
        phi = geo.phi
        c_v_o = params.c_v_o
        c_v_w = params.c_v_w
        rho_r = params.rho_r
        c_r = params.c_r
        oil_mu = params.oil_mu
        oil_rho = params.oil_rho
        water_mu = params.water_mu
        water_rho = params.water_rho
        rel_perm_o = params.rel_perm_o
        rel_perm_w = params.rel_perm_w
        T_inj = params.T_inj
        
        # Define facet quantities
        n = FacetNormal(V.mesh())

        # Define difference between cell centers
        x = SpatialCoordinate(V.mesh())
        x_func = interpolate(x[0], V)
        y_func = interpolate(x[1], V)
        Delta_h = sqrt(jump(x_func)**2 + jump(y_func)**2)

        # harmonic average for permeability and conductivity
        K_x_facet = conditional(gt(avg(K_x), 0.0), K_x('+')*K_x('-') / avg(K_x), 0.0) 
        K_y_facet = conditional(gt(avg(K_y), 0.0), K_y('+')*K_y('-') / avg(K_y), 0.0) 
        
        K_facet = (K_x_facet*(abs(n[0]('+'))+abs(n[0]('-')))/2 + K_y_facet*(abs(n[1]('+'))+abs(n[1]('-')))/2)
        
        kT_facet = conditional(gt(avg(kT), 0.0), kT('+')*kT('-') / avg(kT), 0.0)   
        
        # conservation of mass equation WATER - "pressure equation"
        a_accum_w = phi*(water_rho(p,T)*(1.0 - S_o) - water_rho(p_,T_)*(1.0 - S_o_))/dt*q*dx
        a_flow_w = K_facet*conditional(gt(jump(p), 0.0), rel_perm_w(S_o('+'))*water_rho(p('+'),T('+'))/water_mu(T('+')), rel_perm_w(S_o('-'))*water_rho(p('-'),T('-'))/water_mu(T('-')))*jump(q)*jump(p)/Delta_h*dS
        # conservation of mass equation OIL - "saturation equation"
        a_accum_o = phi*(oil_rho(p,T)*S_o - oil_rho(p_,T_)*S_o_)/dt*s*dx
        a_flow_o = K_facet*conditional(gt(jump(p), 0.0), rel_perm_o(S_o('+'))*oil_rho(p('+'),T('+'))/oil_mu(T('+')), rel_perm_o(S_o('-'))*oil_rho(p('-'),T('-'))/oil_mu(T('-')))*jump(s)*jump(p)/Delta_h*dS
        
        ## WEIGHTED SUM - pressure equation
        a_accum_w = c_v_w*a_accum_w + c_v_o*(phi*(oil_rho(p,T)*S_o - oil_rho(p_,T_)*S_o_)/dt*q*dx)
        a_flow_w = c_v_w*a_flow_w + c_v_o*(K_facet*conditional(gt(jump(p), 0.0), rel_perm_o(S_o('+'))*oil_rho(p('+'),T('+'))/oil_mu(T('+')), rel_perm_o(S_o('-'))*oil_rho(p('-'),T('-'))/oil_mu(T('-')))*jump(q)*jump(p)/Delta_h*dS)
        
        a_Eaccum = phi*c_v_w*(water_rho(p,T)*(1.0 - S_o)*T - water_rho(p_,T_)*(1.0 - S_o_)*T_)/dt*r*dx + phi*c_v_o*(oil_rho(p,T)*S_o*T - oil_rho(p_,T_)*S_o_*T_)/dt*r*dx + (1-phi)*rho_r*c_r*(T - T_)/dt*r*dx 
        a_advec = K_facet*conditional(gt(jump(p), 0.0), T('+')*rel_perm_w(S_o('+'))*water_rho(p('+'),T('+'))/water_mu(T('+')), T('-')*rel_perm_w(S_o('-'))*water_rho(p('-'),T('-'))/water_mu(T('-')))*c_v_w*jump(r)*jump(p)/Delta_h*dS + K_facet*conditional(gt(jump(p), 0.0), T('+')*rel_perm_o(S_o('+'))*oil_rho(p('+'),T('+'))/oil_mu(T('+')), T('-')*rel_perm_o(S_o('-'))*oil_rho(p('-'),T('-'))/oil_mu(T('-')))*c_v_o*jump(r)*jump(p)/Delta_h*dS
        a_diff = kT_facet*jump(T)/Delta_h*jump(r)*dS

        
        F = a_accum_w + a_flow_w + a_Eaccum + a_diff + a_advec
        
        
        rhow_o = oil_rho(p, T)
        rhow_w = water_rho(p, T)
        rhow = water_rho(p, T_inj)
        ## Source terms using global deltas
        if case.name.startswith("Sources"):
            # production wells
            [rate, water_rate, oil_rate] = case.flow_rate_twophase_prod(p, T, S_o = S_o)
            tmp_o = case.deltas_prod*oil_rate
            tmp_w = case.deltas_prod*water_rate
            F -= c_v_w*rhow_w*tmp_w*q*dx + rhow_o*tmp_o*s*dx + c_v_o*rhow_o*tmp_o*q*dx 
            F -= rhow_w*tmp_w*c_v_w*T*r*dx + rhow_o*tmp_o*c_v_o*T*r*dx
            # injection wells
            inj_rate = case.flow_rate_inj(p, T, phase = 'water')
            tmp = case.deltas_inj*inj_rate
            F -= c_v_w*rhow*tmp*q*dx # WEIGHTED SUM
            F -= rhow*tmp*c_v_w*T_inj*r*dx
            # heaters
            F -= case.deltas_heaters*params.U*(T_inj-T)*r*dx
        # Source terms with local deltas
        for well in case.prod_wells:
            [rate, water_rate, oil_rate] = case.flow_rate_twophase(p, T, well, S_o = S_o)
            tmp_o =  well['delta']*oil_rate
            tmp_w = well['delta']*water_rate
            #self.F -= rhow_w*tmp_w*q*dx + rhow_o*tmp_o*s*dx
            F -= c_v_w*rhow_w*tmp_w*q*dx + c_v_o*rhow_o*tmp_o*q*dx # WEIGHTED SUM
            F -= rhow_w*tmp_w*c_v_w*T*r*dx + rhow_o*tmp_o*c_v_o*T*r*dx
        for well in case.inj_wells:
            rate = case.flow_rate(p, T, well, phase = 'water') # only inject water
            tmp = well['delta']*rate
            #self.F -= rhow*tmp*q*dx
            F -= c_v_w*rhow*tmp*q*dx # WEIGHTED SUM
            F -= rhow*tmp*c_v_w*T_inj*r*dx
        for heater in case.heaters:
            tmp = heater['delta']
            F -= tmp*params.U*(T_inj-T)*r*dx
        
        # Getting the pressure temperature sub blocks. 
        J = derivative(F, u)
        a = ExtractSubBlock().split(J, ([0,1], [0,1]))
        
        opts = PETSc.Options()
        self.mat_type = opts.getString(prefix+"schur_mat_type", parameters["default_matrix_type"])
        
        self.A = allocate_matrix(a, bcs=[],form_compiler_parameters=None,
                                  mat_type=self.mat_type)
        self._assemble_A = create_assembly_callable(a, tensor=self.A, bcs=[],
                                                     form_compiler_parameters=None,
                                                     mat_type=self.mat_type)
        self._assemble_A()
        self.A.force_evaluation()

        self.Atildepp = self.A.petscmat
        # Weighted Sum is done in the weak form


        pc_schur = PETSc.PC().create(comm=pc.comm)
        pc_schur.incrementTabLevel(1, parent=pc)
        #pc_schur.setType("lu")
        #pc_schur.setType("hypre")
        #pc_schur.setType("ksp")
        pc_schur.setOptionsPrefix(prefix + "cpr_stage1_")
        pc_schur.setFromOptions()
        pc_schur.setOperators(self.Atildepp, self.Atildepp)
        
        pc_schur.setFieldSplitIS(("0", self.p_is), ("1", self.t_is))
        
        pc_schur.setUp()

        _, k2 = pc_schur.getFieldSplitSubKSP()
        k2.pc.setDM(pc.getDM())
        self.pc_schur = pc_schur        
        
    def update(self, pc):
        self._assemble_A()
        self.A.force_evaluation()
    
    def apply(self, pc, x, y):
        # x is the input Vec to this preconditioner
        # y is the output Vec, P^{-1} x
        x_s = x.getSubVector(self.s_is)
        x_p = x.getSubVector(self.pt_is)
        y_p = y.getSubVector(self.pt_is)
        r_p = x_p.copy()
        
        self.pc_schur.apply(r_p, y_p)
        
        # need to do something for y_s
        y_s = y.getSubVector(self.s_is)
        y_s.set(0.0)

# should not be used
    applyTranspose = apply
     

class CPTRStage1_originalPC(PCBase):
    '''
    1st stage solver for constrained pressure-temperature residual
    '''
    def initialize(self, pc):
        
        prefix = pc.getOptionsPrefix()

        A, P = pc.getOperators()
        appctx = self.get_appctx(pc)
        
        V = appctx["geo"].V
        W = V*V*V
        
        
        fdofs = W.dof_dset.field_ises  # TODO: should this be local_ises?
        self.s_is = fdofs[-1]
        self.pt_is = PETSc.IS().createGeneral(np.concatenate([iset.indices for iset in fdofs[:-1]]))
        self.p_is = fdofs[0]
        self.t_is = fdofs[1]
        
        App = A.createSubMatrix(self.pt_is, self.pt_is)
                              
        self.Atildepp = App # Weighted Sum is done in the weak form

        pc_schur = PETSc.PC().create()
        #pc_schur.setType("lu")
        #pc_schur.setType("hypre")
        #pc_schur.setType("ksp")
        pc_schur.setOptionsPrefix(prefix + "cpr_stage1_")
        pc_schur.setFromOptions()
        pc_schur.setOperators(self.Atildepp)
        
        
        if pc_schur.getType() == "fieldsplit":
            pc_schur.setFieldSplitIS(("0", self.p_is), ("1", self.t_is))
            pc_schur.setUp()
            _, k2 = pc_schur.getFieldSplitSubKSP()
            k2.pc.setDM(pc.getDM())
        elif pc_schur.getType() == "ksp":
            pc_schur.getKSP().setOptionsPrefix(prefix + "cpr_stage1_ksp_")
            pc_schur.getKSP().setFromOptions()
            pc_schur.getKSP().getPC().setFieldSplitIS(("0", self.p_is), ("1", self.t_is))
            pc_schur.setUp()
            _, k2 = pc_schur.getKSP().getPC().getFieldSplitSubKSP()
            k2.pc.setDM(pc.getDM())
        
        self.pc_schur = pc_schur

    def update(self, pc):
        A, P = pc.getOperators()
        App = A.createSubMatrix(self.pt_is, self.pt_is)              
        self.Atildepp = App 
        self.pc_schur.setOperators(self.Atildepp)
    
    def apply(self, pc, x, y):
        # x is the input Vec to this preconditioner
        # y is the output Vec, P^{-1} x
        x_s = x.getSubVector(self.s_is)
        x_p = x.getSubVector(self.pt_is)
        y_p = y.getSubVector(self.pt_is)
        r_p = x_p.copy()
        
        self.pc_schur.apply(r_p, y_p)
        
        # need to do something for y_s
        y_s = y.getSubVector(self.s_is)
        y_s.set(0.0)

    # should not be used
    applyTranspose = apply
