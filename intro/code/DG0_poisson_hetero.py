from firedrake import *
mesh = UnitSquareMesh(20, 20, quadrilateral = True)
V = FunctionSpace(mesh, "DG", 0)
bc = DirichletBC(V, Constant(0), [1, 2], method = "geometric")
x,y = mesh.coordinates
f = interpolate(10*exp(-((x - 0.5)**2 \
    + (y - 0.5)**2) / 0.02), V)
g = interpolate(sin(5*x), V)
x_func = interpolate(x, V)
y_func = interpolate(y, V)
Delta_h = sqrt(jump(x_func)**2 + jump(y_func)**2)

kappa = interpolate(x + y, V)
kappa_facet = conditional(gt(avg(kappa), 0.0), \
              kappa('+')*kappa('-')/avg(kappa), 0.0)     

u = TrialFunction(V)
v = TestFunction(V)
a = kappa_facet*jump(u)/Delta_h*jump(v)*dS
L = f*v*dx + g*v*ds

u = Function(V)
solve(a == L, u, bc)

outfile = File("u_hetero.pvd")
outfile.write(u)
