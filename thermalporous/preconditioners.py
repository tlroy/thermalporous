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
 

class CPRStage1PC(PCBase):
    '''
    1st stage solver for constrained pressure residual
    '''
    def initialize(self, pc):
        from firedrake.formmanipulation import ExtractSubBlock
        from firedrake.assemble import allocate_matrix, create_assembly_callable

        prefix = pc.getOptionsPrefix()
        appctx = self.get_appctx(pc)
        _, P = pc.getOperators()
        
        V = appctx["geo"].V
        if "saturation_space" in appctx:
            W = V*V*V
            second_is = (2)
        else:
            W = V*V
            second_is = (1)
            
        fdofs = W.dof_dset.field_ises 
        self.nonp_is = PETSc.IS().createGeneral(np.concatenate([iset.indices for iset in fdofs[1:]])) # non-pressure fields: saturations and temperatures in the two-phase case
        self.s_is = fdofs[-1] # in singlephase: temperature. in multiphase: saturations
        #self.s_is = self.nonp_is
        self.T_is = fdofs[1] # temperature dofs
        self.p_is = fdofs[0]
        
        
        test = TestFunction(W)
        trial = TrialFunction(W)
        (a, bcs)  = self.form(pc, test, trial)
        splitter = ExtractSubBlock()
        
        # For two-phase: QI/TI only decouples saturations
        # while QI_temp/TI_temp decouples boths sats and temp
        self.decoup = appctx["decoup"]
        
        app = splitter.split(a, ((0), (0)))
        
        # might need to do something about bcs
        opts = PETSc.Options()
        self.mat_type = "aij" #opts.getString(prefix+"schur_mat_type", parameters["default_matrix_type"])
        
        self.App = allocate_matrix(app, form_compiler_parameters=None,
                                  mat_type=self.mat_type)
        self._assemble_App = create_assembly_callable(app, tensor=self.App,
                                                     form_compiler_parameters=None,
                                                     mat_type=self.mat_type)
        self._assemble_App()
        self.App.force_evaluation()

        Appmat = self.App.petscmat
        Appmat.setNullSpace(P.getNullSpace())
        tnullsp = P.getTransposeNullSpace()
        if tnullsp.handle != 0:
            Appmat.setTransposeNullSpace(tnullsp)
        
        self.Appmat = Appmat
        
        if self.decoup == "QI" or self.decoup == "TI":
            aps = splitter.split(a, ((0), second_is))
            self.Aps = allocate_matrix(aps, form_compiler_parameters=None,
                                        mat_type=self.mat_type)
            self._assemble_Aps = create_assembly_callable(aps, tensor=self.Aps,
                                                            form_compiler_parameters=None,
                                                            mat_type=self.mat_type)
            self._assemble_Aps()
            self.Aps.force_evaluation()

            Apsmat = self.Aps.petscmat
            Apsmat.setNullSpace(P.getNullSpace())
            tnullsp = P.getTransposeNullSpace()
            if tnullsp.handle != 0:
                Apsmat.setTransposeNullSpace(tnullsp)
            
            self.Apsmat = Apsmat
            
            asp = splitter.split(a, (second_is, (0) ))
            self.Asp = allocate_matrix(asp, form_compiler_parameters=None,
                                        mat_type=self.mat_type)
            self._assemble_Asp = create_assembly_callable(asp, tensor=self.Asp,
                                                            form_compiler_parameters=None,
                                                            mat_type=self.mat_type)
            self._assemble_Asp()
            self.Asp.force_evaluation()

            Aspmat = self.Asp.petscmat
            Aspmat.setNullSpace(P.getNullSpace())
            tnullsp = P.getTransposeNullSpace()
            if tnullsp.handle != 0:
                Aspmat.setTransposeNullSpace(tnullsp)
            
            self.Aspmat = Aspmat
            
            ass = splitter.split(a, (second_is, second_is))
            self.Ass = allocate_matrix(ass, form_compiler_parameters=None,
                                        mat_type=self.mat_type)
            self._assemble_Ass = create_assembly_callable(ass, tensor=self.Ass,
                                                            form_compiler_parameters=None,
                                                            mat_type=self.mat_type)
            self._assemble_Ass()
            self.Ass.force_evaluation()

            Assmat = self.Ass.petscmat
            Assmat.setNullSpace(P.getNullSpace())
            tnullsp = P.getTransposeNullSpace()
            if tnullsp.handle != 0:
                Assmat.setTransposeNullSpace(tnullsp)
            
            self.Assmat = Assmat
        
        if self.decoup.endswith("temp"):
            aps = splitter.split(a, ((0), (1,2)))
            self.Aps = allocate_matrix(aps, form_compiler_parameters=None,
                                        mat_type=self.mat_type)
            self._assemble_Aps = create_assembly_callable(aps, tensor=self.Aps,
                                                            form_compiler_parameters=None,
                                                            mat_type=self.mat_type)
            self._assemble_Aps()
            self.Aps.force_evaluation()

            Apsmat = self.Aps.petscmat
            Apsmat.setNullSpace(P.getNullSpace())
            tnullsp = P.getTransposeNullSpace()
            if tnullsp.handle != 0:
                Apsmat.setTransposeNullSpace(tnullsp)
            
            self.Apsmat = Apsmat
            
            asp = splitter.split(a, ((1,2), (0)) )
            self.Asp = allocate_matrix(asp, form_compiler_parameters=None,
                                        mat_type=self.mat_type)
            self._assemble_Asp = create_assembly_callable(asp, tensor=self.Asp,
                                                            form_compiler_parameters=None,
                                                            mat_type=self.mat_type)
            self._assemble_Asp()
            self.Asp.force_evaluation()

            Aspmat = self.Asp.petscmat
            Aspmat.setNullSpace(P.getNullSpace())
            tnullsp = P.getTransposeNullSpace()
            if tnullsp.handle != 0:
                Aspmat.setTransposeNullSpace(tnullsp)
            
            self.Aspmat = Aspmat
            
            ass = splitter.split(a, ((1,2), (1,2)))
            self.Ass = allocate_matrix(ass, form_compiler_parameters=None,
                                        mat_type=self.mat_type)
            self._assemble_Ass = create_assembly_callable(ass, tensor=self.Ass,
                                                            form_compiler_parameters=None,
                                                            mat_type=self.mat_type)
            self._assemble_Ass()
            self.Ass.force_evaluation()

            Assmat = self.Ass.petscmat
            Assmat.setNullSpace(P.getNullSpace())
            tnullsp = P.getTransposeNullSpace()
            if tnullsp.handle != 0:
                Assmat.setTransposeNullSpace(tnullsp)
            
            self.Assmat = Assmat
            
            aSS = splitter.split(a, ((2), (2)))
            self.ASS = allocate_matrix(aSS, form_compiler_parameters=None,
                                        mat_type=self.mat_type)
            self._assemble_ASS = create_assembly_callable(aSS, tensor=self.ASS,
                                                            form_compiler_parameters=None,
                                                            mat_type=self.mat_type)
            self._assemble_ASS()
            self.ASS.force_evaluation()

            ASSmat = self.ASS.petscmat
            ASSmat.setNullSpace(P.getNullSpace())
            tnullsp = P.getTransposeNullSpace()
            if tnullsp.handle != 0:
                ASSmat.setTransposeNullSpace(tnullsp)
            
            self.ASSmat = ASSmat
            
            aST = splitter.split(a, ((2), (1)) )
            self.AST = allocate_matrix(aST, form_compiler_parameters=None,
                                        mat_type=self.mat_type)
            self._assemble_AST = create_assembly_callable(aST, tensor=self.AST,
                                                            form_compiler_parameters=None,
                                                            mat_type=self.mat_type)
            self._assemble_AST()
            self.AST.force_evaluation()

            ASTmat = self.AST.petscmat
            ASTmat.setNullSpace(P.getNullSpace())
            tnullsp = P.getTransposeNullSpace()
            if tnullsp.handle != 0:
                ASTmat.setTransposeNullSpace(tnullsp)
            
            self.ASTmat = ASTmat
            
            aTS = splitter.split(a, ((1), (2)))
            self.ATS = allocate_matrix(aTS, form_compiler_parameters=None,
                                        mat_type=self.mat_type)
            self._assemble_ATS = create_assembly_callable(aTS, tensor=self.ATS,
                                                            form_compiler_parameters=None,
                                                            mat_type=self.mat_type)
            self._assemble_ATS()
            self.ATS.force_evaluation()

            ATSmat = self.ATS.petscmat
            ATSmat.setNullSpace(P.getNullSpace())
            tnullsp = P.getTransposeNullSpace()
            if tnullsp.handle != 0:
                ATSmat.setTransposeNullSpace(tnullsp)
            
            self.ATSmat = ATSmat
            
            aTT = splitter.split(a, ((1), (1)))
            self.ATT = allocate_matrix(aTT, form_compiler_parameters=None,
                                        mat_type=self.mat_type)
            self._assemble_ATT = create_assembly_callable(aTT, tensor=self.ATT,
                                                            form_compiler_parameters=None,
                                                            mat_type=self.mat_type)
            self._assemble_ATT()
            self.ATT.force_evaluation()

            ATTmat = self.ATT.petscmat
            ATTmat.setNullSpace(P.getNullSpace())
            tnullsp = P.getTransposeNullSpace()
            if tnullsp.handle != 0:
                ATTmat.setTransposeNullSpace(tnullsp)
            
            self.ATTmat = ATTmat
            
            apS = splitter.split(a, ((0), (2)))
            self.ApS = allocate_matrix(apS, form_compiler_parameters=None,
                                        mat_type=self.mat_type)
            self._assemble_ApS = create_assembly_callable(apS, tensor=self.ApS,
                                                            form_compiler_parameters=None,
                                                            mat_type=self.mat_type)
            self._assemble_ApS()
            self.ApS.force_evaluation()

            ApSmat = self.ApS.petscmat
            ApSmat.setNullSpace(P.getNullSpace())
            tnullsp = P.getTransposeNullSpace()
            if tnullsp.handle != 0:
                ApSmat.setTransposeNullSpace(tnullsp)
            
            self.ApSmat = ApSmat
            
            apT = splitter.split(a, ((0), (1)))
            self.ApT = allocate_matrix(apT, form_compiler_parameters=None,
                                        mat_type=self.mat_type)
            self._assemble_ApT = create_assembly_callable(apT, tensor=self.ApT,
                                                            form_compiler_parameters=None,
                                                            mat_type=self.mat_type)
            self._assemble_ApT()
            self.ApT.force_evaluation()

            ApTmat = self.ApT.petscmat
            ApTmat.setNullSpace(P.getNullSpace())
            tnullsp = P.getTransposeNullSpace()
            if tnullsp.handle != 0:
                ApTmat.setTransposeNullSpace(tnullsp)
            
            self.ApTmat = ApTmat
        
        if self.decoup == "TI":
            self.create_decoup = self.create_decoup_TI
            self.assemble_blocks = self.assemble_blocks_1
        elif self.decoup == "TI_temp":
            self.create_decoup = self.create_decoup_TI_temp
            self.assemble_blocks = self.assemble_blocks_2
        elif self.decoup == "QI":
            self.create_decoup = self.create_decoup_QI
            self.assemble_blocks = self.assemble_blocks_1
        elif self.decoup == "QI_temp":
            self.create_decoup = self.create_decoup_QI_temp
            self.assemble_blocks = self.assemble_blocks_2
        elif self.decoup == "No":
            self.create_decoup = self.create_decoup_
            self.assemble_blocks = self.assemble_blocks_
        self.create_decoup(pc)

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
    
    def form(self, pc, test, trial):
        _, P = pc.getOperators()
        if P.getType() == "python":
            context = P.getPythonContext()
            return (context.a, context.row_bcs)
        else:
            from firedrake.dmhooks import get_appctx
            context = get_appctx(pc.getDM())
            return (context.Jp or context.J, context._problem.bcs)
    
    def assemble_blocks_(self):
        self._assemble_App()
        self.App.force_evaluation()
        
    def assemble_blocks_1(self):
        self._assemble_App()
        self.App.force_evaluation()
        self._assemble_Aps()
        self.Aps.force_evaluation()
        self._assemble_Asp()
        self.Asp.force_evaluation()
        self._assemble_Ass()
        self.Ass.force_evaluation()
    
    def assemble_blocks_2(self):
        self._assemble_App()
        self.App.force_evaluation()
        self._assemble_Aps()
        self.Aps.force_evaluation()
        self._assemble_Asp()
        self.Asp.force_evaluation()
        self._assemble_Ass()
        self.Ass.force_evaluation()
        self._assemble_ASS()
        self.ASS.force_evaluation()
        self._assemble_AST()
        self.AST.force_evaluation()
        self._assemble_ATS()
        self.ATS.force_evaluation()
        self._assemble_ATT()
        self.ATT.force_evaluation()
        self._assemble_ApS()
        self.ApS.force_evaluation()
        self._assemble_ApT()
        self.ApT.force_evaluation()
        

    def create_decoup_(self, pc):
        self.Atildepp = self.Appmat
        pass
    
    def create_decoup_TI(self, pc):

        App = self.Appmat
                              
        # Decoupling operators        
        Ass = self.Assmat
        Asp = self.Aspmat
        Aps = self.Apsmat
        
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
        

    def create_decoup_TI_temp(self, pc):
        import numpy as np

        App = self.Appmat
                              
        # Decoupling operators        
        Ass = self.Assmat
        Asp = self.Aspmat
        Aps = self.Apsmat
        indices = Aps.getOwnershipRange()
        
        ASS = self.ASSmat
        AST = self.ASTmat
        ATS = self.ATSmat
        ATT = self.ATTmat
        ApS = self.ApSmat
        ApT = self.ApTmat
        
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
        
    def create_decoup_QI(self, pc):

        App = self.Appmat
                              
        # Decoupling operators        
        Ass = self.Assmat
        Asp = self.Aspmat
        Aps = self.Apsmat
        
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
        
    def create_decoup_QI_temp(self, pc):
        import numpy as np

        App = self.Appmat
                              
        # Decoupling operators        
        Ass = self.Assmat
        Asp = self.Aspmat
        Aps = self.Apsmat
        indices = Aps.getOwnershipRange()
        
        ASS = self.ASSmat
        AST = self.ASTmat
        ATS = self.ATSmat
        ATT = self.ATTmat
        ApS = self.ApSmat
        ApT = self.ApTmat
        
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
        self.assemble_blocks()
        self.create_decoup(pc)
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

class CPRStage1PC_mat(PCBase):
    '''
    1st stage solver for constrained pressure residual
    '''
    def initialize(self, pc):
        from petsc4py import PETSc

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
        
        self.decoup = appctx["decoup"]        
        
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
        from firedrake.formmanipulation import ExtractSubBlock
        from firedrake.assemble import allocate_matrix, create_assembly_callable
        
        prefix = pc.getOptionsPrefix()

        _, P = pc.getOperators()
        appctx = self.get_appctx(pc)
        
        V = appctx["geo"].V
        W = V*V*V
        W22 = V*V
        
        test = TestFunction(W)
        trial = TrialFunction(W)
        (a, bcs)  = self.form(pc, test, trial)
        splitter = ExtractSubBlock()
        app = splitter.split(a, ((0, 1), (0, 1)))
        
        # might need to do something about bcs
        opts = PETSc.Options()
        self.mat_type = opts.getString(prefix+"schur_mat_type", parameters["default_matrix_type"])
        
        self.App = allocate_matrix(app, form_compiler_parameters=None,
                                  mat_type=self.mat_type)
        self._assemble_App = create_assembly_callable(app, tensor=self.App,
                                                     form_compiler_parameters=None,
                                                     mat_type=self.mat_type)
        self._assemble_App()
        self.App.force_evaluation()

        Appmat = self.App.petscmat
        Appmat.setNullSpace(P.getNullSpace())
        tnullsp = P.getTransposeNullSpace()
        if tnullsp.handle != 0:
            Appmat.setTransposeNullSpace(tnullsp)
        
        self.Atildepp = Appmat # Weighted Sum is done in the weak form
        
        pc_schur = PETSc.PC().create()
        #pc_schur.setType("lu")
        #pc_schur.setType("hypre")
        #pc_schur.setType("ksp")
        pc_schur.setOptionsPrefix(prefix + "cpr_stage1_")
        pc_schur.setFromOptions()
        pc_schur.setOperators(self.Atildepp)
        
        if pc_schur.getType() == "fieldsplit":
            pc_schur.setDM(W22.dm)
            pc_schur.setUp()
            _, k2 = pc_schur.getFieldSplitSubKSP()
            k2.pc.setDM(pc.getDM())
        elif pc_schur.getType() == "ksp":
            pc_schur.getKSP().setOptionsPrefix(prefix + "cpr_stage1_ksp_")
            pc_schur.getKSP().setFromOptions()
            pc_schur.setDM(W22.dm)
            pc_schur.setUp()
            _, k2 = pc_schur.getKSP().getPC().getFieldSplitSubKSP()
            k2.pc.setDM(pc.getDM())
        
        self.pc_schur = pc_schur
        
        fdofs = W.dof_dset.field_ises 
        self.s_is = fdofs[-1]
        self.pt_is = PETSc.IS().createGeneral(np.concatenate([iset.indices for iset in fdofs[:-1]]))
        self.p_is = fdofs[0]
        self.t_is = fdofs[1]   
        
    def form(self, pc, test, trial):
        _, P = pc.getOperators()
        if P.getType() == "python":
            context = P.getPythonContext()
            return (context.a, context.row_bcs)
        else:
            from firedrake.dmhooks import get_appctx
            context = get_appctx(pc.getDM())
            return (context.Jp or context.J, context._problem.bcs)

    def update(self, pc):
        self._assemble_App()
        self.App.force_evaluation()
    
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
