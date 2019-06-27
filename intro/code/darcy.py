from firedrake import *
from thermalporous.physicalparameters import PhysicalParameters as Params
import thermalporous.utils as utils
params = Params()
Nx = 40
Ny = 40
L = 50.0
mesh = SquareMesh(Nx, Ny, L, quadrilateral = True)
V = FunctionSpace(mesh, "DQ", 0)
bc = DirichletBC(V, Constant(0), [1, 2], method = "geometric")
x,y = mesh.coordinates
#K_x = Constant(3e-7)
#K_y = Constant(3e-7)
K_x = Function(V).interpolate(1e-5*(1e-5+ (y/L)**2*1.1**(-abs((L*0.4)**2-((x-0.5*L)**2+(y-0.5*L)**2)))))
K_y = Function(V).interpolate(1e-5*(1e-5+ 10.0*(x/L)*(y/L)**2*1.1**(-abs((L*0.4)**2-((x-0.5*L)**2+(y-0.5*L)**2)))))
File("results/permeability_x.pvd").write(K_x)
File("results/permeability_y.pvd").write(K_y)
phi = Constant(0.2)
T = params.T_ref
rho = lambda p : params.oil_rho(p, T)
mu = Constant(params.oil_mu(T))
initial_condition = Constant(params.p_ref)
dt = Constant(3600.0*24.0*5.0)

# Initiate functions
p = Function(V)
p_ = Function(V)
q = TestFunction(V)

# Define facet quantities
n = FacetNormal(mesh)

# Define difference between cell centers
x_func = interpolate(x, V)
y_func = interpolate(y, V)
Delta_h = sqrt(jump(x_func)**2 + jump(y_func)**2)

# harmonic average for permeability and conductivity
K_x_facet = conditional(gt(avg(K_x), 0.0), K_x('+')*K_x('-') / avg(K_x), 0.0) 
K_y_facet = conditional(gt(avg(K_y), 0.0), K_y('+')*K_y('-') / avg(K_y), 0.0) 
K_facet = (K_x_facet*(abs(n[0]('+'))+abs(n[0]('-')))/2 \ 
          +  K_y_facet*(abs(n[1]('+'))+abs(n[1]('-')))/2) 

# conservation of mass equation
a_accum = phi*(rho(p) - rho(p_))/dt*q*dx
a_flow = K_facet/mu*conditional(gt(jump(p), 0.0), rho(p('+')), rho(p('-'))) \
         *jump(q)*jump(p)/Delta_h*dS
F = a_accum + a_flow 

# Source terms
def well_delta(w):
    delta = Function(V)
    node_w = utils.GetNodeClosestToCoordinate(V, w)
    vec = delta.vector().get_local()
    if node_w >= 0:
        vec[node_w] = 1.0
    delta.vector().set_local(vec)
    normalise = L**2/float(Nx)/float(Ny)
    return delta.assign(delta/normalise)

prod_well = well_delta((5.0,L/2.0))
inj_well = well_delta((L-5.0,L/2.0))

prod_rate = -1e-6
F -= prod_rate*prod_well*rho(p)*q*dx

inj_rate = 1e-6
F -= inj_rate*inj_well*rho(p)*q*dx

## Solution
outfile = File("results/pressure.pvd")
p_.assign(initial_condition)
p.assign(initial_condition)
outfile.write(p)
for i in range(20):
    solve(F == 0, p)
    p_.assign(p)
    outfile.write(p)


