import numpy as np

from firedrake import *

#import thermalporous.utils as utils


class ThermalModel:
        
    def __init__(self, end = 1.0, maxdt = 0.005, save = False, n_save = 2, small_dt_start = True, checkpointing = {}, filename = "results/results.txt", dt_init_fact = 2**(-10)):

        self.maxdt = maxdt
        self.dt_init_fact = dt_init_fact
        self.dt = Constant(maxdt*24.0*3600.0)
        self.end = end # in days
        self.init_variational_form()
        self.init_solver_parameters()
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
        solver = NonlinearVariationalSolver(self.problem, appctx=self.appctx, solver_parameters=self.solver_parameters)
        
        #snes = solver.snes
        #if snes.ksp.pc.getType() == 'composite' and self.solver_parameters["mat_type"] == "aij": 
            #print("using aij CPR")
            #subpc = snes.ksp.pc.getCompositePC(0)
            ##subpcksp = snes.ksp.pc.getCompositePC(0).getKSP()
            ##subpc = subpcksp.getPC()
            #subpc.setType('python')
            #cpr_stage1 = self.create_cpr_stage_1(snes)
            #subpc.setPythonContext(cpr_stage1)
                   
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
        end = self.end*24.0*3600.0
        #end = dt*nt # Final time
        
        t = 0.0
        dt_counter = 0 #for average iteration count
        total_lits = 0
        total_nits = 0
        
        if self.save:
            if self.vector:
                outfileu = File("results/vecsolution.pvd")
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
        if self.comm.rank == 0:
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
        
        while (t < end):
            i += 1
            previous_fail = 0
            if self.comm.rank == 0:
                print("Time: ", t/24.0/3600.0, " days. Time-step ", i, ". dt size: ", self.dt.values()[0]/24.0/3600.0)  
                
                
            if self.case.inj_wells:    #in cases with many wells, calculate individual rates is slow with assemble. 
                current_inj_rate_ = 0 
                for well in self.case.inj_wells:
                    current_inj_rate_ += well['delta']*well['rate']*dx
                current_inj_rate = assemble(current_inj_rate_)
                if self.comm.rank == 0:
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
                    if self.comm.rank == 0:
                        print("Total water production rate is ", current_water_rate)
                        print("Total oil production rate is ", current_oil_rate)
                if self.comm.rank == 0:
                    print("Total production rate is ", current_prod_rate,
                    #"\nPressure: ", pvec.vector()[well['node']], ". Temperature: ", Tvec.vector()[well['node']], 
                    )
            if False:
                for well in self.case.inj_wells:
                    current_inj_rate = assemble(well['delta']*well['rate']*dx)
                    if self.comm.rank == 0:
                        print("New injection rate for well ", well['name'], " is ", current_inj_rate#,
                        #"\nPressure: ", pvec.vector()[well['node']], ". Temperature: ", Tvec.vector()[well['node']]
                        )
                
                for well in self.case.prod_wells:
                    current_prod_rate = assemble(well['delta']*well['rate']*dx)
                    if self.name == 'Two-phase':
                        current_water_rate = assemble(well['delta']*well['water_rate']*dx)
                        current_oil_rate = assemble(well['delta']*well['oil_rate']*dx)
                        if self.comm.rank == 0:
                            print("Water rate for well ", well['name'], " is ", current_water_rate)
                            print("Oil rate for well ", well['name'], " is ", current_oil_rate)
                    if self.comm.rank == 0:
                        print("New production rate for well ", well['name'], " is ", current_prod_rate,
                        #"\nPressure: ", pvec.vector()[well['node']], ". Temperature: ", Tvec.vector()[well['node']], 
                        )
                    
            if self.comm.rank == 0:
                print(" ")
                    
            while True:
                try:
                    self.solver.solve()
                    if self.name == "Two-phase":
                        print(np.max(u.dat.data[2]))
                except exceptions.ConvergenceError:
                    print(np.max(u.dat.data[2]))
                    #if self.name == "Two-phase":
                        #(pvec, Tvec, S_ovec) = u.split()
                        #outfileS_o.write(S_ovec)
                    #else:
                        #(pvec, Tvec) = u.split()
                    if self.comm.rank == 0:
                        print(i_plot, "th plot")
                    #outfilep.write(pvec)
                    #outfileT.write(Tvec)
                    #from IPython import embed; embed()
                    self.dt.assign(self.dt.values()[0]*0.5)
                    print("Time: ", t/24.0/3600.0, " days. Time-step ", i, ". New dt size: ", self.dt.values()[0]/24.0/3600.0)  
                    u.assign(u_)
                    previous_fail = 1
                    continue
                break
            # making sure 0<= S_o <=1
            if self.name == "Two-phase":
                
                Sat = u.dat.data[2]
                #epsilon = 0.0
                while False: #np.amax(Sat)- 1.0 > epsilon or np.amin(Sat) < -epsilon: #
                    print("------Negative saturation! Chopping time-step---------")
                    self.dt.assign(self.dt.values()[0]*0.5)
                    print("Time: ", t/24.0/3600.0, " days. Time-step ", i, ". New dt size: ", self.dt.values()[0]/24.0/3600.0)  
                    u.assign(u_)
                    self.solver.solve()
                    Sat = u.dat.data[2]
                N = len(Sat)
                Sat = np.minimum(Sat, np.ones(int(N)))
                Sat = np.maximum(Sat, np.zeros(int(N)))
                u.dat.data[2][...] = Sat    
                

            
            u_.assign(u)
            current_dt = self.dt.values()[0] 
            t += current_dt
            dt_vec.append(current_dt)
            
            
            # save solutions
            if self.save:
                if i_plot%self.n_save == 0:
                    if self.vector:
                        outfileu.write(u)
                    else:
                        if self.name == "Two-phase":
                            (pvec, Tvec, S_ovec) = u.split()
                            outfileS_o.write(S_ovec)
                        else:
                            (pvec, Tvec) = u.split()
                        if self.comm.rank == 0:
                            print(i_plot, "th plot")
                        outfilep.write(pvec)
                        outfileT.write(Tvec)
                i_plot += 1

            
            # update time-step
            
            
            #if self.comm.rank == 0:
            current_nits = self.solver.snes.getIterationNumber() # can use this for adaptive time-step
            current_lits = self.solver.snes.getLinearSolveIterations()
            if self.comm.rank == 0:
                print("Nonlinear iterations: ", current_nits)
                print("Linear iterations: ", current_lits)
                total_nits += current_nits
                total_lits += current_lits
                dt_counter += 1
                nits_vec.append(current_nits)
                lits_vec.append(current_lits)
            if self.geo.name.startswith("SPE10"):
                if current_nits < 6:# and previous_fail == 0:
                    factor = 1 + min(1.0, (6 - current_nits)**2/3**2)
                    self.dt.assign(min(dt_inj,current_dt*factor))
                elif current_nits > 9:
                    factor = 1 -  min(1.0, (current_nits - 9)**2/4**2)/2
                    self.dt.assign(current_dt*factor)
                else:
                    self.dt.assign(current_dt)
            current_dt = self.dt.values()[0] 
            if current_dt > end-t:
                self.dt.assign(end-t)
            
                    
         
        # Save final solution for convergence test
        self.u = u
        
        if self.checkpointing["save"] is True:
            chk = DumbCheckpoint(self.checkpointing["savename"], mode=FILE_CREATE)
            chk.store(u, "solution")
            self.resultprint("Saving checkpoint solution in " + self.checkpointing["savename"])
        
        # convergence of iterative methods at each time-step
        if self.comm.rank == 0:
            self.resultprint("nits = ", nits_vec, ";")
            self.resultprint("lits = ", lits_vec, ";")
            self.resultprint("dts = ", dt_vec, ";")
            
            # Model details
            self.resultprint("----------------------------------------------------------------------")
            self.resultprint(self.name, "thermal model")
            self.resultprint("Geo model: ", self.geo.name)
            self.resultprint("Test case: ", self.case.name)
            self.resultprint("Max time-step: ", self.maxdt)
            self.resultprint("Final time: ", t/24.0/3600.0)#(t-self.dt.values()[0])/24.0/3600.0)
                
            # Print results
            nits = self.solver.snes.getIterationNumber()
            lits = self.solver.snes.getLinearSolveIterations()

            self.resultprint("Solver Parameters")
            self.resultprint("-----------------")
            for x in self.solver_parameters:
                self.resultprint(x, ':', self.solver_parameters[x])
            if self.solver.snes.ksp.pc.getType() == 'fieldsplit':
                self.resultprint(self.idorder)
            self.resultprint(" ")
            self.resultprint("Solver performance")
            self.resultprint("------------------")
            avg_nitdt = total_nits/dt_counter
            avg_litdt = total_lits/dt_counter
            avg_litnit = avg_litdt/avg_nitdt
            self.resultprint("Average Nonlinear iterations per time-step:", avg_nitdt)
            self.resultprint("Average Linear iterations per time-step: ", avg_litdt)
            self.resultprint("Average Linear iteration per Nonlinear iteration: ", avg_litnit)
            self.resultprint("Total Linear iterations: ", sum(lits_vec))
            self.resultprint("Number of time-steps: ", len(dt_vec))
            self.resultprint("Last Nonlinear iterations:", nits)
            self.resultprint("Last Linear iterations: ", lits)
            self.resultprint("----------------------------------------------------------------------")
            self.resultprint(" ")
            
            self.total_nits = total_nits
            self.total_lits = total_lits
            self.last_dt = dt_vec[-1]
    
