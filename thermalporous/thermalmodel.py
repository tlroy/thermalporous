import numpy as np

from firedrake import *
import thermalporous.utils as utils
from datetime import datetime

class ThermalModel:
        
    def __init__(self, end = 1.0, maxdt = 0.005, save = False, n_save = 2, small_dt_start = True, checkpointing = {}, filename = "results/results.txt", dt_init_fact = 2**(-10), verbosity = True):

        self.maxdt = maxdt
        self.dt_init_fact = dt_init_fact
        self.dt = Constant(maxdt*24.0*3600.0)
        self.end = end # in days
        self.verbosity = verbosity
        #self.init_solver_parameters()
        self.init_variational_form()
        self.init_solver()
        
        self.checkpointing = {"save": False, "load": False, "savename": "initial", "loadname": "initial"}
        self.checkpointing.update(checkpointing)
        self.filename = filename
        try:
            self.initial_condition = self.case.init_IC(phases = self.name)
        except AttributeError:
            self.initial_condition = self.init_IC_uniform()
            
        if filename == "results/results.txt":
            opens = "a" # add
        else:
            opens = "w" # overwrite
        self.f = open(self.filename, opens)
        
    def init_solver(self):
        
        self.problem = NonlinearVariationalProblem(self.F, self.u, self.bcs)
        if not self.verbosity:
            del self.solver_parameters["snes_monitor"]
            del self.solver_parameters["snes_converged_reason"]
            del self.solver_parameters["ksp_converged_reason"]

        solver = NonlinearVariationalSolver(self.problem, appctx=self.appctx, solver_parameters=self.solver_parameters)
        
        if "ksp_monitor_residuals" in self.solver_parameters:
            res = Function(self.W)
            if self.name == "Two-phase":
                def my_monitor(ksp, its, rnorm):
                    residual = ksp.buildResidual() 
                    with res.dat.vec_wo as v:
                        residual.copy(v)
                    vec_p = res.sub(0).vector()
                    rnorm_p = sqrt(vec_p.inner(vec_p))
                    vec_E = res.sub(1).vector()
                    rnorm_E = sqrt(vec_E.inner(vec_E))
                    vec_o = res.sub(2).vector()
                    rnorm_o = sqrt(vec_o.inner(vec_o))
                    if self.comm.rank == 0 and self.verbosity:
                        print("  --> Pressure equation residual: %s" % (rnorm_p))
                        print("  --> Energy equation residual: %s" % (rnorm_E))
                        print("  --> Oil equation residual: %s" % (rnorm_o))
            else:
                def my_monitor(ksp, its, rnorm):
                    residual = ksp.buildResidual() 
                    with res.dat.vec_wo as v:
                        residual.copy(v)
                    vec_M = res.sub(0).vector()
                    rnorm_M = sqrt(vec_M.inner(vec_M))
                    vec_E = res.sub(1).vector()
                    rnorm_E = sqrt(vec_E.inner(vec_E))
                    if self.comm.rank == 0 and self.verbosity:
                        print("  --> Mass equation residual: %s" % (rnorm_M))
                        print("  --> Energy equation residual: %s" % (rnorm_E))
                        
            solver.snes.ksp.setMonitor(my_monitor)
                   
        self.solver = solver

    def resultprint(self, *output):
        print(*output)
        print(*output, file = self.f)
        
    def solve(self):
                      
        u = self.u
        u_ = self.u_
        # Initial condition 
        if self.checkpointing["load"] is True:
            self.resultprint("Using as initial solution checkpoint " + self.checkpointing["loadname"])
            chk = DumbCheckpoint(self.checkpointing["loadname"], mode=FILE_READ)
            chk.load(u, name = "solution")
            chk.load(u_, name = "solution")
        else:
            u.assign(self.initial_condition)
            u_.assign(self.initial_condition)
        
        # Time-stepping parameters
        dt_init = self.dt_init_fact*self.maxdt*24.0*3600.0
        dt_inj = self.maxdt*24.0*3600.0
        dt_prod = self.maxdt*24.0*3600.0
        dt_rest = 10.0*self.maxdt*24.0*3600.0
        if self.small_dt_start:
            self.dt.assign(dt_init) # size of steps
        t_init = 1.0*24.0*3600.0
        t_rest = 50.0*24.0*3600.0
        t_prod = 0.0*24.0*3600.0
        end = self.end*24.0*3600.0 # Final time
        
        t = 0.0
        dt_counter = 0 #for average iteration count
        total_lits = 0
        total_nits = 0
        
        if self.save:
            if self.vector:
                outfileu = File("results/vecsolution.pvd")
                if self.name == "Two-phase":
                    outfileS_o = File("results/saturation_o.pvd")
                    (pTvec, S_ovec) = u.split()
                    outfileS_o.write(S_ovec)
                    outfileu.write(pTvec)
                else:
                    outfileu.write(u)
            else:
                outfilep = File("results/pressure.pvd")
                outfileT = File("results/temperature.pvd")
                if self.name == "Two-phase":
                    outfileS_o = File("results/saturation_o.pvd")
                    (pvec, Tvec, S_ovec) = u.split()
                    outfileS_o.write(S_ovec)
                else:
                    (pvec, Tvec) = u.split()
                outfilep.write(pvec)
                outfileT.write(Tvec)
        if self.comm.rank == 0 and self.verbosity:
            print("Solving time-dependent problem")

        init_true = 0
        inj_true = 0
        prod_true = 0
        prod_rate = 0
        i = 0
        i_plot = 0
        nits_vec = []
        lits_vec = []
        dt_vec = []
        epsilon = np.finfo(float).eps
        timings = []

        #start_cpu = datetime.now()
        #old_cpu = start_cpu
        while (t < end):
            i += 1
            previous_fail = 0
            if self.comm.rank == 0 and self.verbosity:
                print("Time: ", t/24.0/3600.0, " days. Time-step ", i, ". dt size: ", self.dt.values()[0]/24.0/3600.0, flush = True)  
                
            
                    
            if self.comm.rank == 0 and self.verbosity:
                print(" ")
                    
            while True:
                try:
                    old_cpu = datetime.now()
                    self.solver.solve() 
                    now_cpu = datetime.now()
                    timings.append((now_cpu-old_cpu).total_seconds())
                    if self.name == "Two-phase":
                        print(np.max(u.dat.data[self.i_S_o]))
                except exceptions.ConvergenceError:
                    if self.name == "Two-phase":
                        print(np.max(u.dat.data[self.i_S_o]))
                    if self.comm.rank == 0 and self.verbosity:
                        print(i_plot, "th plot")
                    self.dt.assign(self.dt.values()[0]*0.5)
                    if self.comm.rank == 0 and self.verbosity:
                        print("Time: ", t/24.0/3600.0, " days. Time-step ", i, ". New dt size: ", self.dt.values()[0]/24.0/3600.0, flush = True)  
                    u.assign(u_)
                    previous_fail = 1
                    continue
                break
            
            # making sure 0<= S_o <=1
            if self.name == "Two-phase":
                if self.vector:
                    (pT,S_o) = split(u)
                    (p,T) = split(pT)
                else:
                    (p,T,S_o) = split(u)
                mass_o = assemble(self.geo.phi*S_o*self.params.oil_rho(p,T)*dx)
                if self.comm.rank == 0 and self.verbosity:
                    print("Total oil mass in reservoir: ", mass_o, flush = True)
                Sat = u.dat.data[self.i_S_o]
                epsilon = 1e-10
                local_chop = (np.amax(Sat)- 1.0 > epsilon or np.amin(Sat) < -epsilon,)
                from mpi4py import MPI
                global_chop = (False,)
                global_chop = self.comm.reduce(local_chop, op=MPI.MAX, root=0)
                global_chop = self.comm.bcast(global_chop, root=0)
                
                while global_chop[0]: #False: #
                    if self.comm.rank == 0 and self.verbosity:
                        print("------Negative saturation! Chopping time-step---------", flush = True)
                    self.dt.assign(self.dt.values()[0]*0.5)
                    if self.comm.rank == 0 and self.verbosity:
                        print("Time: ", t/24.0/3600.0, " days. Time-step ", i, ". New dt size: ", self.dt.values()[0]/24.0/3600.0, flush = True)  
                    u.assign(u_)
                    try:
                        self.solver.solve()
                    except exceptions.ConvergenceError:
                        print(np.max(u.dat.data[self.i_S_o]))
                        self.dt.assign(self.dt.values()[0]*0.5)
                        if self.comm.rank == 0 and self.verbosity:
                            print("Time: ", t/24.0/3600.0, " days. Time-step ", i, ". New dt size: ", self.dt.values()[0]/24.0/3600.0, flush = True)  
                        u.assign(u_)
                        previous_fail = 1
                        continue
                    break
                    Sat = u.dat.data[self.i_S_o]
                    local_chop = (np.amax(Sat)- 1.0 > epsilon or np.amin(Sat) < -epsilon,)
                    global_chop = (False,)
                    global_chop = self.comm.reduce(local_chop, op=MPI.MAX, root=0)
                    global_chop = self.comm.bcast(global_chop, root=0)
                    
                    
                N = len(Sat)
                Sat = np.minimum(Sat, np.ones(int(N)))
                Sat = np.maximum(Sat, np.zeros(int(N)))
                u.dat.data[self.i_S_o][...] = Sat    
                
            ## Print prod/inj rates    
            if self.case.name.startswith("Sources"):
                current_inj_rate = assemble(self.case.deltas_inj*self.inj_rate*dx)
                if self.comm.rank == 0 and self.verbosity:
                    print("Total injection rate is ", current_inj_rate)
                current_prod_rate = assemble(self.case.deltas_prod*self.prod_rate*dx)
                if self.comm.rank == 0 and self.verbosity:
                    print("Total production rate is ", current_prod_rate)
                if self.name == "Two-phase":
                    current_oil_rate = assemble(self.case.deltas_prod*self.oil_rate*dx)
                    if self.comm.rank == 0 and self.verbosity:
                        print("Total oil production rate is ", current_oil_rate)
                    current_water_rate = assemble(self.case.deltas_prod*self.water_rate*dx)
                    if self.comm.rank == 0 and self.verbosity:
                        print("Total water production rate is ", current_water_rate)

            if self.case.inj_wells:    #in cases with many wells, calculate individual rates is slow with assemble. In those cases, might be better to use SourceTerms instead
                current_inj_rate_ = 0 
                for well in self.case.inj_wells:
                    current_inj_rate_ += well['delta']*well['rate']*dx
                current_inj_rate = assemble(current_inj_rate_)
                if self.comm.rank == 0 and self.verbosity:
                    print("Total injection rate is ", current_inj_rate#,
                    #"\nPressure: ", pvec.vector()[well['node']], ". Temperature: ", Tvec.vector()[well['node']]
                    )
            if self.case.prod_wells:
                current_prod_rate_ = 0
                for well in self.case.prod_wells:
                    current_prod_rate_ += well['delta']*well['rate']*dx
                current_prod_rate = assemble(current_prod_rate_)
                if self.name == 'Two-phase':
                    current_water_rate_ = 0; current_oil_rate_ = 0
                    for well in self.case.prod_wells:
                        current_water_rate_ += well['delta']*well['water_rate']*dx
                        current_oil_rate_ += well['delta']*well['oil_rate']*dx
                    current_water_rate = assemble(current_water_rate_)
                    current_oil_rate = assemble(current_oil_rate_)
                    if self.comm.rank == 0 and self.verbosity:
                        print("Total water production rate is ", current_water_rate)
                        print("Total oil production rate is ", current_oil_rate)
                if self.comm.rank == 0 and self.verbosity:
                    print("Total production rate is ", current_prod_rate,
                    #"\nPressure: ", pvec.vector()[well['node']], ". Temperature: ", Tvec.vector()[well['node']], 
                    )
            if False:
                for well in self.case.inj_wells:
                    current_inj_rate = assemble(well['delta']*well['rate']*dx)
                    if self.comm.rank == 0 and self.verbosity:
                        print("New injection rate for well ", well['name'], " is ", current_inj_rate#,
                        #"\nPressure: ", pvec.vector()[well['node']], ". Temperature: ", Tvec.vector()[well['node']]
                        )
                
                for well in self.case.prod_wells:
                    current_prod_rate = assemble(well['delta']*well['rate']*dx)
                    if self.name == 'Two-phase':
                        current_water_rate = assemble(well['delta']*well['water_rate']*dx)
                        current_oil_rate = assemble(well['delta']*well['oil_rate']*dx)
                        if self.comm.rank == 0 and self.verbosity:
                            print("Water rate for well ", well['name'], " is ", current_water_rate)
                            print("Oil rate for well ", well['name'], " is ", current_oil_rate)
                    if self.comm.rank == 0 and self.verbosity:
                        print("New production rate for well ", well['name'], " is ", current_prod_rate,
                        #"\nPressure: ", pvec.vector()[well['node']], ". Temperature: ", Tvec.vector()[well['node']], 
                        )    

            u_.assign(u)
            #self.b = assemble(self.F)
            current_dt = self.dt.values()[0] 
            t += current_dt
            dt_vec.append(current_dt)
            
            
            # save solutions
            if self.save:
                if i_plot%self.n_save == 0:
                    if self.vector:
                        if self.name == "Two-phase":
                            (pTvec, S_ovec) = u.split()
                            outfileS_o.write(S_ovec)
                            outfileu.write(pTvec)
                        else:
                            outfileu.write(u)
                    else:
                        if self.name == "Two-phase":
                            (pvec, Tvec, S_ovec) = u.split()
                            outfileS_o.write(S_ovec)
                        else:
                            (pvec, Tvec) = u.split()
                        if self.comm.rank == 0 and self.verbosity:
                            print(i_plot, "th plot")
                        outfilep.write(pvec)
                        outfileT.write(Tvec)
                i_plot += 1

            
            # update time-step
            current_nits = self.solver.snes.getIterationNumber() # can use this for adaptive time-step
            current_lits = self.solver.snes.getLinearSolveIterations()
            if self.comm.rank == 0 and self.verbosity:
                print("Nonlinear iterations: ", current_nits)
                print("Linear iterations: ", current_lits)
                total_nits += current_nits
                total_lits += current_lits
                dt_counter += 1
                nits_vec.append(current_nits)
                lits_vec.append(current_lits)
            if self.geo.name.startswith("SPE10"): # Simple heuristic for an adaptive time-step
                if current_nits < 6:# and previous_fail == 0:
                    factor = 1 + min(1.0, (6 - current_nits)**2/3**2)
                    self.dt.assign(min(dt_inj,current_dt*factor))
                elif current_nits > 9:
                    factor = 1 -  min(1.0, (current_nits - 9)**2/4**2)/2
                    self.dt.assign(current_dt*factor)
                else:
                    self.dt.assign(current_dt)
            current_dt = self.dt.values()[0] 
            if current_dt > end-t and t < end:
                self.dt.assign(end-t)
            #now_cpu = datetime.now()
            #timings.append((now_cpu-old_cpu).total_seconds())
            #old_cpu = now_cpu

        end_cpu = datetime.now()
        # Save final solution for convergence test
        #self.u.assign(u)
        #self.u_.assign(u_)
        #self.dt.assign(current_dt)
        #utils.ExportJacobian(self.J, u)
        #utils.ExportResidual(self.F)
        
        if self.checkpointing["save"] is True:
            chk = DumbCheckpoint(self.checkpointing["savename"], mode=FILE_CREATE)
            chk.store(u, "solution")
            self.resultprint("Saving checkpoint solution in " + self.checkpointing["savename"])
        
        # convergence of iterative methods at each time-step
        if self.comm.rank == 0 and self.verbosity:
            self.resultprint("nits = ", nits_vec, ";")
            self.resultprint("lits = ", lits_vec, ";")
            self.resultprint("dts = ", dt_vec, ";")
            self.resultprint("timings = ", timings, ";")
            
            # Model details
            self.resultprint("----------------------------------------------------------------------")
            self.resultprint(self.name, "thermal model")
            self.resultprint("Geo model: ", self.geo.name)
            self.resultprint("Test case: ", self.case.name)
            self.resultprint("Max time-step: ", self.maxdt)
            self.resultprint("Final time: ", t/24.0/3600.0)
                
            # Print results
            nits = self.solver.snes.getIterationNumber()
            lits = self.solver.snes.getLinearSolveIterations()

            self.resultprint("Solver Parameters")
            self.resultprint("-----------------")
            for x in self.solver_parameters:
                self.resultprint(x, ':', self.solver_parameters[x])
            if self.solver.snes.ksp.pc.getType() == 'fieldsplit' and self.name == "Single phase":
                self.resultprint(self.idorder)
            self.resultprint(" ")
            self.resultprint("Solver performance")
            self.resultprint("------------------")
            #self.resultprint("Total CPU time (s):", (end_cpu-start_cpu).total_seconds())
            self.resultprint("Total CPU time (s):", sum(timings))
            avg_nitdt = total_nits/dt_counter
            avg_litdt = total_lits/dt_counter
            avg_litnit = avg_litdt/avg_nitdt
            self.resultprint("Average Nonlinear iterations per time-step:", avg_nitdt)
            self.resultprint("Average Linear iterations per time-step: ", avg_litdt)
            self.resultprint("Average Linear iteration per Nonlinear iteration: ", avg_litnit)
            self.resultprint("Total Linear iterations: ", sum(lits_vec))
            self.resultprint("Total Nonlinear iterations: ", sum(nits_vec))
            self.resultprint("Number of time-steps: ", len(dt_vec))
            self.resultprint("Last Nonlinear iterations:", nits)
            self.resultprint("Last Linear iterations: ", lits)
            self.resultprint("----------------------------------------------------------------------")
            self.resultprint(" ")
            
            self.total_nits = total_nits
            self.total_lits = total_lits
            self.last_dt = dt_vec[-1]
    
