from firedrake import *
mesh = UnitSquareMesh(20, 20) 
V = FunctionSpace(mesh, "CG", 1)
bc = DirichletBC(V, Constant(0), [1, 2])
x,y = mesh.coordinates
f = interpolate(10*exp(-((x - 0.5)**2 \
    + (y - 0.5)**2) / 0.02), V)
g = interpolate(sin(5*x), V)

u = TrialFunction(V)
v = TestFunction(V)
a = inner(grad(u), grad(v))*dx
L = f*v*dx + g*v*ds

u = Function(V)
solve(a == L, u, bc)

outfile = File("u_poissonCG.pvd")
outfile.write(u)
