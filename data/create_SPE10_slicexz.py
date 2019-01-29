# load data for SPE10. must be run in serial

from firedrake import *
import numpy as np

Nx = 100
Nz = 20

def create_SPE10_slicexz(Nx, Nz, x_shift = 0, y_shift = 0, z_shift = 0, perm_factor = 1.0):
    import os
    dirname = os.path.dirname(__file__)

    Dx = 6.096
    Dz = 0.6096

    #Length_x = Nx*Dx
    #Length_z = Nz*Dz

    #mesh = RectangleMesh(Nx, Nz, Length_x, Length_z, quadrilateral=True)
    #VDG = FunctionSpace(mesh, "DQ", 0)



    lines = np.loadtxt(dirname + "/spe_phi.dat")

    #k = 0
    #single_line = [0]*Nx*Ny*Nz
    #print(len(single_line))
    #for kk in range(0, Nz):
        #for j in range(0,int(Nx*Ny/6)):
            #for i in range(0,6):
                #single_line[k] = lines[j+kk*2200,i]
                #k += 1       
    single_line = np.reshape(lines, 60*220*85)
    #k = 0
    field = np.zeros((Nx,Nz))
    for kk in range(0,Nz):
        for i in range(0,Nx):
            field[i][kk] = single_line[(i+x_shift)+(y_shift)*60+(kk+z_shift)*220*60]

                
    phi_array = np.array(field)

    np.save(dirname + '/slice_phi.npy', phi_array)

    # load data for permeability
    lines = np.loadtxt(dirname + "/spe_perm.dat")
    #perm_factor = 1.0e0
    print("Using SPE10 permeability/porosity fields")
    print("Permeability factor is: ", perm_factor)
    lines =  lines*9.869233e-10*perm_factor #converting md to mm^2  

    single_line = np.reshape(lines, [3, 60*220*85]).transpose()
    #k = 0
    field = np.zeros((Nx,Nz))
    for kk in range(0,Nz):
        for i in range(0,Nx):
            field[i][kk] = single_line[(i+x_shift)+(y_shift)*60+(kk+z_shift)*220*60,0]
    
    permx_array = np.array(field)

    np.save(dirname + '/slice_perm_x.npy', permx_array)

    field = np.zeros((Nx,Nz))
    for kk in range(0,Nz):
        for i in range(0,Nx):
            field[i][kk] = single_line[(i+x_shift)+(y_shift)*60+(kk+z_shift)*220*60,2]
    
    permy_array = np.array(field)

    np.save(dirname + '/slice_perm_y.npy', permy_array)

def create_SPE10_sliceyz(Ny, Nz, x_shift = 0, y_shift = 0, z_shift = 0, perm_factor = 1.0):
    import os
    dirname = os.path.dirname(__file__)

    #Dy = 3.048
    #Dz = 0.6096

    #Length_y = Ny*Dy
    #Length_z = Nz*Dz

    #meshbase = RectangleMesh(Nx, Ny, Length_x, Length_y, quadrilateral=True)
    #mesh = ExtrudedMesh(meshbase, Nz, Dz)
    #VDG = FunctionSpace(mesh, "DQ", 0)



    lines = np.loadtxt(dirname + "/spe_phi.dat")

    single_line = np.reshape(lines, 60*220*85)
    #k = 0
    field = np.zeros((Ny,Nz))
    for kk in range(0,Nz):
        for j in range(0,Ny):
            field[j][kk] = single_line[(x_shift)+(j+y_shift)*60+(kk+z_shift)*220*60]
                #k += 1
                
    phi_array = np.array(field)

    np.save(dirname + '/slice_phi.npy', phi_array)

    # load data for permeability
    lines = np.loadtxt(dirname + "/spe_perm.dat")
    print("Using SPE10 permeability/porosity fields")
    print("Permeability factor is: ", perm_factor)
    lines =  lines*9.869233e-10*perm_factor #converting md to mm^2  

    single_line = np.reshape(lines, [3, 60*220*85]).transpose()
    #k = 0
    field = np.zeros((Ny,Nz))
    for kk in range(0,Nz):
        for j in range(0,Ny):
            field[j][kk] = single_line[(x_shift)+(j+y_shift)*60+(kk+z_shift)*220*60,1]
    
    permx_array = np.array(field)

    np.save(dirname + '/slice_perm_x.npy', permx_array)

    for kk in range(0,Nz):
        for j in range(0,Ny):
            field[j][kk] = single_line[(x_shift)+(j+y_shift)*60+(kk+z_shift)*220*60,2]
    
    permy_array = np.array(field)

    np.save(dirname + '/slice_perm_y.npy', permy_array)



