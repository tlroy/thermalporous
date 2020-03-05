import numpy as np

from firedrake import *

from firedrake_citations import Citations
from firedrake.petsc import PETSc

import firedrake.dmhooks as dmhooks
from firedrake.dmhooks import get_function_space, get_appctx, push_appctx

class ConvDiffSchurPC(PCBase):
    def initialize(self, pc):
        from firedrake import TrialFunction, TestFunction, dx, assemble, inner, parameters
        from firedrake.assemble import allocate_matrix, create_assembly_callable
        prefix = pc.getOptionsPrefix()

        # we assume P has things stuffed inside of it
        _, P = pc.getOperators()

        appctx = self.get_appctx(pc)
        fcp = appctx.get("form_compiler_parameters")

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
            
            if geo.gravity2D:
                g = params.g
                flow = jump(p0)/Delta_h - g*avg(oil_rho(p0,T0))*(abs(n[1]('+'))+abs(n[1]('-')))/2
            else:
                flow = jump(p0)/Delta_h
                
            # conservation of energy equation
            a_Eaccum = phi*c_v*(oil_rho(p0,T0)*T)/dt*r*dx + (1-phi)*rho_r*c_r*(T)/dt*r*dx
            a_advec = K_facet*conditional(gt(flow, 0.0), T('+')*oil_rho(p0('+'),T0('+'))/oil_mu(T0('+')), T('-')*oil_rho(p0('-'),T0('-'))/oil_mu(T0('-')))*c_v*jump(r)*flow*dS
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

        Pmat = self.A.petscmat
        Pmat.setNullSpace(P.getNullSpace())
        tnullsp = P.getTransposeNullSpace()
        if tnullsp.handle != 0:
            Pmat.setTransposeNullSpace(tnullsp)

        from firedrake.variational_solver import NonlinearVariationalProblem
        from firedrake.solving_utils import _SNESContext
        dm = pc.getDM()
        octx = get_appctx(dm)
        oproblem = octx._problem
        nproblem = NonlinearVariationalProblem(oproblem.F, oproblem.u, bcs=octx._problem.bcs, J=octx.J, form_compiler_parameters=fcp)
        self._ctx_ref = _SNESContext(nproblem, self.mat_type, self.mat_type, octx.appctx)


        ksp = PETSc.KSP().create()
        ksp.incrementTabLevel(1, parent=pc)
        ksp.getPC().setDM(dm)
        ksp.setOperators(Pmat)
        ksp.setOptionsPrefix(prefix + "schur_")
        
        with dmhooks.add_hooks(dm, self, appctx=self._ctx_ref, save=False):
            ksp.setFromOptions()
        with dmhooks.add_hooks(dm, self, appctx=self._ctx_ref):
            ksp.setUp()
        
        #from IPython import embed; embed()
        self.ksp = ksp

    def update(self, pc):
        self._assemble_A()

    def apply(self, pc, X, Y):
        dm = pc.getDM()
        with dmhooks.add_hooks(dm, self, appctx=self._ctx_ref):
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
        fcp = appctx.get("form_compiler_parameters")

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
            if geo.gravity2D:
                g = params.g
                flow_o = jump(p0)/Delta_h - g*avg(oil_rho(p0,T0))*(abs(n[1]('+'))+abs(n[1]('-')))/2
                flow_w = jump(p0)/Delta_h - g*avg(water_rho(p0,T0))*(abs(n[1]('+'))+abs(n[1]('-')))/2
            else:
                flow_o = jump(p0)/Delta_h
                flow_w = jump(p0)/Delta_h
            # conservation of energy equation
            a_Eaccum = phi*c_v_o*S0*(oil_rho(p0,T0)*T)/dt*r*dx + phi*c_v_w*(1.0 - S0)*(water_rho(p0,T0)*T)/dt*r*dx + (1-phi)*rho_r*c_r*(T)/dt*r*dx 
            a_advec = K_facet*conditional(gt(flow_w, 0.0), T('+')*rel_perm_w(S0('+'))*water_rho(p0('+'),T0('+'))/water_mu(T0('+')), T('-')*rel_perm_w(S0('-'))*water_rho(p0('-'),T0('-'))/water_mu(T0('-')))*c_v_w*jump(r)*flow_w*dS + K_facet*conditional(gt(flow_o, 0.0), T('+')*rel_perm_o(S0('+'))*oil_rho(p0('+'),T0('+'))/oil_mu(T0('+')), T('-')*rel_perm_o(S0('-'))*oil_rho(p0('-'),T0('-'))/oil_mu(T0('-')))*c_v_o*jump(r)*flow_o*dS
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

        self.A = allocate_matrix(a, form_compiler_parameters=fcp,
                                  mat_type=self.mat_type)
        self._assemble_A = create_assembly_callable(a, tensor=self.A,
                                                     form_compiler_parameters=fcp,
                                                     mat_type=self.mat_type)
        self._assemble_A()

        Pmat = self.A.petscmat
        Pmat.setNullSpace(P.getNullSpace())
        tnullsp = P.getTransposeNullSpace()
        if tnullsp.handle != 0:
            Pmat.setTransposeNullSpace(tnullsp)

        from firedrake.variational_solver import NonlinearVariationalProblem
        from firedrake.solving_utils import _SNESContext
        dm = pc.getDM()
        octx = get_appctx(dm)
        oproblem = octx._problem
        nproblem = NonlinearVariationalProblem(oproblem.F, oproblem.u, bcs=octx._problem.bcs, J=octx.J, form_compiler_parameters=fcp)
        self._ctx_ref = _SNESContext(nproblem, self.mat_type, self.mat_type, octx.appctx)
        
        

        ksp = PETSc.KSP().create()
        ksp.incrementTabLevel(1, parent=pc)
        ksp.getPC().setDM(dm)
        ksp.setOperators(Pmat)
        ksp.setOptionsPrefix(prefix + "schur_")
        
        with dmhooks.add_hooks(dm, self, appctx=self._ctx_ref, save=False):
            ksp.setFromOptions()
        with dmhooks.add_hooks(dm, self, appctx=self._ctx_ref):
            ksp.setUp()
        
        #from IPython import embed; embed()
        self.ksp = ksp

    def update(self, pc):
        self._assemble_A()

    def apply(self, pc, X, Y):
        dm = pc.getDM()
        with dmhooks.add_hooks(dm, self, appctx=self._ctx_ref):
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
        fcp = appctx.get("form_compiler_parameters")
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

            Assmat = self.Ass.petscmat
            Assmat.setNullSpace(P.getNullSpace())
            tnullsp = P.getTransposeNullSpace()
            if tnullsp.handle != 0:
                Assmat.setTransposeNullSpace(tnullsp)

            self.Assmat = Assmat

        if self.decoup.endswith("temp"):
            W22 = V*V
            #fdofs_fields = W22.dof_dset.field_ises 
            fdofs_ss = W22.dof_dset.field_ises 
            #self.fdofs_fields = W22.dof_dset.field_ises 
            #self.fdofs_ss = W22.dof_dset.local_ises 
            self.TT_is = fdofs_ss[0]
            self.SS_is = fdofs_ss[1]
            fdofs_pp = V.dof_dset.field_ises
            self.pp_is = fdofs_pp[0]

            aps = splitter.split(a, ((0), (1,2)))
            self.Aps = allocate_matrix(aps, form_compiler_parameters=None,
                                        mat_type=self.mat_type)
            self._assemble_Aps = create_assembly_callable(aps, tensor=self.Aps,
                                                            form_compiler_parameters=None,
                                                            mat_type=self.mat_type)
            self._assemble_Aps()

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

        from firedrake.variational_solver import NonlinearVariationalProblem
        from firedrake.solving_utils import _SNESContext
        dm = pc.getDM()
        octx = get_appctx(dm)
        oproblem = octx._problem
        nproblem = NonlinearVariationalProblem(oproblem.F, oproblem.u, bcs, J=a, form_compiler_parameters=fcp)
        self._ctx_ref = _SNESContext(nproblem, self.mat_type, self.mat_type, octx.appctx)

        ptx = self._ctx_ref.split(([0], [0]))[0]
        pdm = ptx._x.function_space().dm
        self._pctx_ref = ptx

        pc_schur.setDM(pdm)
        #pc_schur.setType("lu")
        #pc_schur.setType("hypre")
        #pc_schur.setType("ksp")
        pc_schur.setOperators(self.Atildepp)
        pc_schur.setOptionsPrefix(prefix + "cpr_stage1_")

        with dmhooks.add_hooks(pdm, self, appctx=ptx,  save=False):
            pc_schur.setFromOptions()
        with dmhooks.add_hooks(pdm, self, appctx=ptx):
            pc_schur.setUp()

        self.pc_schur = pc_schur

        if not self.decoup == "No":
            psize = self.p_is.getSizes()
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

    def assemble_blocks_1(self):
        self._assemble_App()
        self._assemble_Aps()
        self._assemble_Asp()
        self._assemble_Ass()

    def assemble_blocks_2(self):
        self._assemble_App()
        self._assemble_Aps()
        self._assemble_Asp()
        self._assemble_Ass()
        self._assemble_ASS()
        self._assemble_AST()
        self._assemble_ATS()
        self._assemble_ATT()
        self._assemble_ApS()
        self._assemble_ApT()

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

        ASS = self.ASSmat
        AST = self.ASTmat
        ATS = self.ATSmat
        ATT = self.ATTmat
        ApS = self.ApSmat
        ApT = self.ApTmat
        p_indices = ApT.getOwnershipRange()

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

        block = np.zeros([2,2])

        size = self.SS_is.getLocalSize()
        index_range = range(p_indices[0], p_indices[-1])
        Trange = self.TT_is.array
        Srange = self.SS_is.array

        for i in range(0,size) :
            block[0,0] = Rsum_TT[index_range[i]]
            block[0,1] = Rsum_TS[index_range[i]]
            block[1,0] = Rsum_ST[index_range[i]]
            block[1,1] = Rsum_SS[index_range[i]]
            block = np.linalg.inv(block)
            invDss.setValue(Trange[i], Trange[i], block[0,0])
            invDss.setValue(Trange[i], Srange[i], block[0,1])
            invDss.setValue(Srange[i], Trange[i], block[1,0])
            invDss.setValue(Srange[i], Srange[i], block[1,1])
        invDss.assemblyBegin()
        invDss.assemblyEnd()

        Dps = PETSc.Mat().create()
        Dps.setSizes(Aps.getSizes())
        Dps.setUp()

        Prange = self.pp_is.array

        for i in range(0,size):
            Dps.setValue(Prange[i],Trange[i], Rsum_pT[index_range[i]])
            Dps.setValue(Prange[i],Srange[i], Rsum_pS[index_range[i]])
        Dps.assemblyBegin()
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
        p_indices = Aps.getOwnershipRange()

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

        block = np.zeros([2,2])
        size = self.SS_is.getLocalSize()
        index_range = range(p_indices[0], p_indices[-1])
        Trange = self.TT_is.array
        Srange = self.SS_is.array

        for i in range(0,size) :
            block[0,0] = Rsum_TT[index_range[i]]
            block[0,1] = Rsum_TS[index_range[i]]
            block[1,0] = Rsum_ST[index_range[i]]
            block[1,1] = Rsum_SS[index_range[i]]
            block = np.linalg.inv(block)
            invDss.setValue(Trange[i], Trange[i], block[0,0])
            invDss.setValue(Trange[i], Srange[i], block[0,1])
            invDss.setValue(Srange[i], Trange[i], block[1,0])
            invDss.setValue(Srange[i], Srange[i], block[1,1])

        invDss.assemblyBegin()
        invDss.assemblyEnd()

        Dps = PETSc.Mat().create()
        Dps.setSizes(Aps.getSizes())
        Dps.setUp()

        Prange = self.pp_is.array

        for i in range(0,size):
            Dps.setValue(Prange[i],Trange[i], Rsum_pT[index_range[i]])
            Dps.setValue(Prange[i],Srange[i], Rsum_pS[index_range[i]])
        Dps.assemblyBegin()
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
        #from IPython import embed; embed()
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

        pdm = self.pc_schur.getDM()
        with dmhooks.add_hooks(pdm, self, appctx=self._pctx_ref ):
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
        pc_schur.setOperators(self.Atildepp)
        pc_schur.setOptionsPrefix(prefix + "cpr_stage1_")
        pc_schur.setFromOptions()
        pc_schur.setUp()

        self.pc_schur = pc_schur

        if not self.decoup == "No":
            psize = self.p_is.getSizes()
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
        p_indices = Aps.getOwnershipRange()

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
        #invDss.assemblyBegin()
        block = np.zeros([2,2])
        s_size = self.s_is.getSize()

        for i in range(p_indices[0], p_indices[-1]):
            #print(rank, i, p_indices)
            block[0,0] = Rsum_TT[i]
            block[0,1] = Rsum_TS[i]
            block[1,0] = Rsum_ST[i]
            block[1,1] = Rsum_SS[i]
            block = np.linalg.inv(block)
            invDss.setValue(i,i, block[0,0])
            invDss.setValue(i,i + s_size, block[0,1])
            invDss.setValue(i + s_size,i, block[1,0])
            invDss.setValue(i + s_size,i + s_size, block[1,1])
        invDss.assemblyBegin()
        invDss.assemblyEnd()

        Dps = PETSc.Mat().create()
        Dps.setSizes(Aps.getSizes())
        Dps.setUp()
        for i in range(p_indices[0], p_indices[-1]):
            Dps.setValue(i,i, Rsum_pT[i])
            Dps.setValue(i,i + s_size, Rsum_pS[i])
        Dps.assemblyBegin()
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
        p_indices = Aps.getOwnershipRange()

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
        block = np.zeros([2,2])
        s_size = self.s_is.getSize()

        for i in range(p_indices[0], p_indices[-1]):
            #print(rank, i, p_indices)
            block[0,0] = Rsum_TT[i]
            block[0,1] = Rsum_TS[i]
            block[1,0] = Rsum_ST[i]
            block[1,1] = Rsum_SS[i]
            block = np.linalg.inv(block)
            invDss.setValue(i,i, block[0,0])
            invDss.setValue(i,i + s_size, block[0,1])
            invDss.setValue(i + s_size,i, block[1,0])
            invDss.setValue(i + s_size,i + s_size, block[1,1])
        invDss.assemblyBegin()
        invDss.assemblyEnd()

        p_indices = Aps.getOwnershipRange()
        Dps = PETSc.Mat().create()
        Dps.setSizes(Aps.getSizes())
        Dps.setUp()
        for i in range(p_indices[0], p_indices[-1]):
            Dps.setValue(i,i, Aps.getValue(i,i))
            Dps.setValue(i,i + s_size, Aps.getValue(i,i+s_size))
        Dps.assemblyBegin()
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

        fdofs = W.dof_dset.field_ises
        self.s_is = PETSc.IS().createGeneral(np.concatenate([iset.indices for iset in fdofs[0:2:1]]))
        self.p_is = fdofs[1]

        App = A.createSubMatrix(self.p_is, self.p_is)

        self.Atildepp = App # Weighted Sum is done in the weak form

        pc_schur = PETSc.PC().create()
        #pc_schur.setType("lu")
        #pc_schur.setType("hypre")
        #pc_schur.setType("ksp")

        pc_schur.setOperators(self.Atildepp)
        pc_schur.setOptionsPrefix(prefix + "cpr_stage1_")
        pc_schur.setFromOptions()
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
        fcp = appctx.get("form_compiler_parameters")

        V = appctx["geo"].V
        if appctx["vector"]:
            W22 = VectorFunctionSpace(V.mesh(), "DQ", degree = 0, dim = 2)
            W = W22*V
            primary_is = (0)
            second_is = (1)
        else:
            W = V*V*V
            W22 = V*V
            primary_is = (0,1)
            second_is = (2)

        test = TestFunction(W)
        trial = TrialFunction(W)
        (a, bcs)  = self.form(pc, test, trial)
        splitter = ExtractSubBlock()
        a00 = splitter.split(a, (primary_is, primary_is))

        self.decoup = appctx["decoup"]

        # might need to do something about bcs
        opts = PETSc.Options()
        self.mat_type = opts.getString(prefix+"schur_mat_type", parameters["default_matrix_type"])

        self.A00 = allocate_matrix(a00, form_compiler_parameters=None,
                                  mat_type=self.mat_type)
        self._assemble_A00 = create_assembly_callable(a00, tensor=self.A00,
                                                     form_compiler_parameters=None,
                                                     mat_type=self.mat_type)
        self._assemble_A00()

        A00mat = self.A00.petscmat
        A00mat.setNullSpace(P.getNullSpace())
        tnullsp = P.getTransposeNullSpace()
        if tnullsp.handle != 0:
            A00mat.setTransposeNullSpace(tnullsp)

        self.A00mat = A00mat

        if self.decoup == "QI" or self.decoup == "TI":
            a0s = splitter.split(a, (primary_is, second_is))
            self.A0s = allocate_matrix(a0s, form_compiler_parameters=None,
                                        mat_type=self.mat_type)
            self._assemble_A0s = create_assembly_callable(a0s, tensor=self.A0s,
                                                            form_compiler_parameters=None,
                                                            mat_type=self.mat_type)
            self._assemble_A0s()

            A0smat = self.A0s.petscmat
            A0smat.setNullSpace(P.getNullSpace())
            tnullsp = P.getTransposeNullSpace()
            if tnullsp.handle != 0:
                A0smat.setTransposeNullSpace(tnullsp)
            self.A0smat = A0smat

            as0 = splitter.split(a, (second_is, primary_is ))
            self.As0 = allocate_matrix(as0, form_compiler_parameters=None,
                                        mat_type=self.mat_type)
            self._assemble_As0 = create_assembly_callable(as0, tensor=self.As0,
                                                            form_compiler_parameters=None,
                                                            mat_type=self.mat_type)
            self._assemble_As0()

            As0mat = self.As0.petscmat
            As0mat.setNullSpace(P.getNullSpace())
            tnullsp = P.getTransposeNullSpace()
            if tnullsp.handle != 0:
                As0mat.setTransposeNullSpace(tnullsp)
            self.As0mat = As0mat

            ass = splitter.split(a, (second_is, second_is))
            self.Ass = allocate_matrix(ass, form_compiler_parameters=None,
                                        mat_type=self.mat_type)
            self._assemble_Ass = create_assembly_callable(ass, tensor=self.Ass,
                                                            form_compiler_parameters=None,
                                                            mat_type=self.mat_type)
            self._assemble_Ass()

            Assmat = self.Ass.petscmat
            Assmat.setNullSpace(P.getNullSpace())
            tnullsp = P.getTransposeNullSpace()
            if tnullsp.handle != 0:
                Assmat.setTransposeNullSpace(tnullsp)
            self.Assmat = Assmat

        if self.decoup == "TI":
            self.create_decoup = self.create_decoup_TI
            self.assemble_blocks = self.assemble_blocks_1
        elif self.decoup == "QI":
            self.create_decoup = self.create_decoup_QI
            self.assemble_blocks = self.assemble_blocks_1
        elif self.decoup == "No":
            self.create_decoup = self.create_decoup_
            self.assemble_blocks = self.assemble_blocks_
        self.create_decoup(pc)

        pc_schur = PETSc.PC().create()

        from firedrake.variational_solver import NonlinearVariationalProblem
        from firedrake.solving_utils import _SNESContext
        dm = pc.getDM()
        octx = get_appctx(dm)
        oproblem = octx._problem
        nproblem = NonlinearVariationalProblem(oproblem.F, oproblem.u, bcs, J=a, form_compiler_parameters=fcp)
        self._ctx_ref = _SNESContext(nproblem, self.mat_type, self.mat_type, octx.appctx)

        ptx = self._ctx_ref.split(([0,1], [0,1]))[0]
        pdm = ptx._x.function_space().dm
        self._pctx_ref = ptx

        pc_schur.setDM(pdm)

        #pc_schur.setType("lu")
        #pc_schur.setType("hypre")
        #pc_schur.setType("ksp")
        #pc_schur.setDM(W22.dm)
        pc_schur.setOperators(self.Atilde00)
        pc_schur.setOptionsPrefix(prefix + "cpr_stage1_")
        with dmhooks.add_hooks(pdm, self, appctx = self._pctx_ref, save = False):
            pc_schur.setFromOptions()
        with dmhooks.add_hooks(pdm, self, appctx = self._pctx_ref):
            pc_schur.setUp()


        #if pc_schur.getType() == "fieldsplit":
        #    pc_schur.setDM(W22.dm)
        #    pc_schur.setOptionsPrefix(prefix + "cpr_stage1_")
        #    pc_schur.setFromOptions()
        #    fdofs2 = W22.dof_dset.field_ises
        #    pc_schur.setFieldSplitIS(("0", fdofs2[0]), ("1", fdofs2[1]))
        #    pc_schur.setUp()
        #    _, k2 = pc_schur.getFieldSplitSubKSP()
        #    k2.pc.setDM(pc.getDM())
        #elif pc_schur.getType() == "ksp":
        #    if pc_schur.pc.getType() == "fieldsplit":
        #        pc_schur.getKSP().setOptionsPrefix(prefix + "cpr_stage1_ksp_")
        #        pc_schur.getKSP().setFromOptions()
        #        pc_schur.setDM(W22.dm)
        #        pc_schur.setUp()
        #        _, k2 = pc_schur.getKSP().getPC().getFieldSplitSubKSP()
        #        k2.pc.setDM(pc.getDM())
        #elif pc_schur.getType() == "hypre":
        #    pc_schur.setDM(W22.dm)
        #    pc_schur.setUp()
        #else:
        #    pc_schur.setDM(W22.dm)
        #    pc_schur.setUp()

        self.pc_schur = pc_schur

        fdofs = W.dof_dset.field_ises
        self.s_is = fdofs[-1]
        if appctx["vector"]:
            self.pt_is = fdofs[0]
        else:
            self.pt_is = PETSc.IS().createGeneral(np.concatenate([iset.indices for iset in fdofs[:-1]]))
            #self.p_is = fdofs[0]
            #self.t_is = fdofs[1]

        if not self.decoup == "No":
            ptsize = self.pt_is.getSizes()
            # Vector for storing preconditioned pressure-temperature vector
            self.workVec = PETSc.Vec().create()
            self.workVec.setSizes(ptsize)
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
        self._assemble_A00()

    def assemble_blocks_1(self):
        self._assemble_A00()
        self._assemble_A0s()
        self._assemble_As0()
        self._assemble_Ass()

    def create_decoup_(self, pc):
        self.Atilde00 = self.A00mat

    def create_decoup_TI(self, pc):

        A00 = self.A00mat
        # Decoupling operators
        Ass = self.Assmat
        As0 = self.As0mat
        A0s = self.A0smat
        p_indices = Ass.getOwnershipRange()

        #True-IMPES
        AssT = PETSc.Mat()
        Ass.transpose(out=AssT)
        invdiag = AssT.getRowSum()
        invdiag.reciprocal()
        invDss = PETSc.Mat().create()
        invDss.setSizes(Ass.getSizes())
        invDss.setUp()
        invDss.setDiagonal(invdiag)

        D0s = PETSc.Mat().create()
        D0s.setSizes(A0s.getSizes())
        D0s.setBlockSizes(A0s.getBlockSizes()[0],A0s.getBlockSizes()[1])
        D0s.setUp()

        weightp = PETSc.Vec().create()
        weightp.setSizes(A0s.getSizes()[0], bsize = 2)
        weightp.setUp()
        weightT = PETSc.Vec().create()
        weightT.setSizes(A0s.getSizes()[0], bsize = 2)
        weightT.setUp()
        for i in range(p_indices[0], p_indices[-1]):
            weightp.setValue(2*i, 1.0)
            weightT.setValue(2*i+1, 1.0)
        weightp.assemblyBegin()
        weightp.assemblyEnd()
        weightT.assemblyBegin()
        weightT.assemblyEnd()

        sump = PETSc.Vec().create()
        sump.setSizes(A0s.getSizes()[1])
        sump.setUp()

        sumT = PETSc.Vec().create()
        sumT.setSizes(A0s.getSizes()[1])
        sumT.setUp()

        A0s.multTranspose(weightp,sump)
        A0s.multTranspose(weightT,sumT)

        for i in range(p_indices[0], p_indices[-1]):
            D0s.setValue(2*i,i, sump.getValue(i))
            D0s.setValue(2*i + 1, i, sumT.getValue(i))
        D0s.assemblyBegin()
        D0s.assemblyEnd()

        self.a0sinvdss = D0s.matMult(invDss)

        A00.axpy(-1.0, self.a0sinvdss.matMult(As0), structure = 2) # Weighted Sum is done in the weak form
        self.Atilde00 = A00

    def create_decoup_QI(self, pc):
        A00 = self.A00mat

        # Decoupling operators
        Ass = self.Assmat
        As0 = self.As0mat
        A0s = self.A0smat

        p_indices = Ass.getOwnershipRange()
        #Quasi-IMPES
        invdiag = Ass.getDiagonal()
        invdiag.reciprocal()
        invDss = PETSc.Mat().create()
        invDss.setSizes(Ass.getSizes())
        invDss.setUp()
        invDss.setDiagonal(invdiag)

        #D0s = A0s
        #diag = A0s.getDiagonal()
        #D0s = PETSc.Mat().create()
        #D0s.setSizes(A0s.getSizes())
        #D0s.setUp()
        #D0s.setDiagonal(diag)

        D0s = PETSc.Mat().create()
        D0s.setSizes(A0s.getSizes())
        D0s.setBlockSizes(A0s.getBlockSizes()[0],A0s.getBlockSizes()[1])
        D0s.setUp()

        for i in range(p_indices[0], p_indices[-1]):
            D0s.setValue(2*i,i, A0s.getValue(2*i,i))
            D0s.setValue(2*i + 1, i, A0s.getValue(2*i+1,i))
        D0s.assemblyBegin()
        D0s.assemblyEnd()

        self.a0sinvdss = D0s.matMult(invDss)

        A00.axpy(-1.0, self.a0sinvdss.matMult(As0), structure = 2) # Weighted Sum is done in the weak form
        self.Atilde00 = A00

    def update(self, pc):
        self.assemble_blocks()
        self.create_decoup(pc)
        self.pc_schur.setOperators(self.Atilde00)

    def apply(self, pc, x, y):
        # x is the input Vec to this preconditioner
        # y is the output Vec, P^{-1} x
        x_p = x.getSubVector(self.pt_is)
        y_p = y.getSubVector(self.pt_is)
        r_p = x_p.copy()

        if not self.decoup == "No":
            x_s = x.getSubVector(self.s_is)
            self.a0sinvdss.mult(x_s, self.workVec)
            r_p.axpy(  -1.0, self.workVec)  # x_p - A0s*inv(Dss)*x_s
        pdm = self.pc_schur.getDM()
        with dmhooks.add_hooks(pdm, self, appctx = self._pctx_ref):
            self.pc_schur.apply(r_p, y_p)

        # need to do something for y_s
        y_s = y.getSubVector(self.s_is)
        y_s.set(0.0)


    # should not be used
    applyTranspose = apply
