from firedrake import *

import thermalporous.utils as utils


class HeaterCase():
    
    def __init__(self, params, geo, well_case = None, heater_points = list()):
        self.name = 'Heaters'
        self.Length = geo.Length
        self.Length_y = geo.Length_y
        if geo.dim == 3: 
            self.Length_z = geo.Length_z
        self.V = geo.V
        self.mesh = geo.mesh
        self.geo = geo
        self.params = params
        if self.geo.dim == 2:
            if well_case == "default":
                prod_points = [[0.2*self.Length, self.Length_y/2]]
                inj_points = [[0.8*self.Length, self.Length_y/2]]
            elif well_case == "SPE10_60x120":
                x_inj = 265.0
                y_inj = 260.0
                x_prod = 140.0
                y_prod = 210.0
                prod_points = [[x_prod, y_prod]]
                inj_points = [[x_inj, y_inj]]
                heater_points = prod_points + inj_points
            elif well_case == "test0":
                Length = self.Length
                Length_y = self.Length_y
                prod_points = [[2., Length_y/4.], [2., Length_y/2.], [2., 3.*Length_y/4]] 
                inj_points = [[Length-2., Length_y/4.], [Length-2., Length_y/2.], [Length-2., 3.*Length_y/4]]
                heater_points = prod_points + inj_points
            elif well_case == "test":
                Lx = self.Length
                Ly = self.Length_y
                prod_points = [[10., 0.1*Ly], [10., 0.2*Ly], [10., 0.3*Ly], [10., 0.4*Ly], [10., 0.5*Ly], [10., 0.6*Ly], [10., 0.7*Ly], [10., 0.8*Ly], [10., 0.9*Ly]]
                inj_points = [[Lx - 10., 0.1*Ly], [Lx - 10., 0.2*Ly], [Lx - 10., 0.3*Ly], [Lx - 10., 0.4*Ly], [Lx - 10., 0.5*Ly], [Lx - 10., 0.6*Ly], [Lx - 10., 0.7*Ly], [Lx - 10., 0.8*Ly], [Lx - 10., 0.9*Ly]]
                heater_points = prod_points + inj_points
                
        elif self.geo.dim == 3:
            if well_case == "default":
                prod_points = [[self.Length/2, self.Length_y/2, self.Length_z*0.2]]
                inj_points = [[self.Length/2, self.Length_y/2, self.Length_z*0.8]]
                heater_points = prod_points + inj_points
            if well_case == "multiple":
                prod_points = [[self.Length/4, self.Length_y/2, self.Length_z*0.2], [self.Length/2, self.Length_y/2, self.Length_z*0.2], [3*self.Length/4, self.Length_y/2, self.Length_z*0.2]]
                inj_points = [[self.Length/4, self.Length_y/2, self.Length_z*0.8], [self.Length/2, self.Length_y/2, self.Length_z*0.8], [3*self.Length/4, self.Length_y/2, self.Length_z*0.8]]
                heater_points = prod_points + inj_points
        
        self.init_heaters(heater_points, 'circle')
        
        
    def init_heaters(self, heater_points, wellfunc):
        self.heaters = []
        
        self.heatercount = 0
        for point in heater_points:
            self.heaters.append(self.make_heater(point, wellfunc))
            
    def make_heater(self, w, wellfunc):
        if wellfunc == 'delta':
            delta = self.well_delta(w)
        elif wellfunc == 'circle':
            if self.geo.dim == 2:
                delta = self.well_circle(w)
            elif self.geo.dim == 3:
                delta = self.well_circle3D(w)
        
        current_count = str(self.heatercount)
        self.heatercount += 1

        return {'name': 'heater' + current_count, 'location': w,
                #'node': node,
                'delta': delta}
        
        
    def well_circle(self, w):
        xw = w[0]
        yw = w[1]
        x, y = SpatialCoordinate(self.mesh)
        radius = 0.1 #0.1875*0.3048 # 0.1 radius of well # 0.1875*0.3048 from ChenZhang2009
        delta = Function(self.V)
        delta.assign(interpolate(conditional(pow(x-xw,2)+pow(y-yw,2)<pow(radius,2), exp(-(1.0/(-pow(x-xw,2)-pow(y-yw,2)+pow(radius,2)))), 0.0), self.V))
        normalise = assemble(delta*dx)
        if normalise == 0:  
            delta = self.well_delta(w)
        else:
            delta.assign(delta/normalise)
        return delta
    
    def well_circle3D(self, w):
        xw = w[0]
        yw = w[1]
        zw = w[2]
        x, y, z = SpatialCoordinate(self.mesh)
        radius = 0.1 #0.1875*0.3048 # 0.1 radius of well # 0.1875*0.3048 from ChenZhang2009
        height = 1.0
        delta = Function(self.V)
        delta.assign(interpolate(conditional(pow(x-xw,2)+pow(y-yw,2)<pow(radius,2), conditional(abs(z-zw)<height, 1.0, 0.0)*exp(-(1.0/(-pow(x-xw,2)-pow(y-yw,2)+pow(radius,2)))), 0.0), self.V))
        normalise = assemble(delta*dx)
        if normalise == 0:  
            delta = self.well_delta(w)
        else:
            delta.assign(delta/normalise)
        return delta
        
    def well_delta(self, w):
        delta = Function(self.V)
        node_w = utils.GetNodeClosestToCoordinate(self.V, w)
        vec = delta.vector().get_local()
        if node_w >= 0:
            vec[node_w] = 1.0
        delta.vector().set_local(vec)
        normalise = assemble(delta*dx)
        return delta.assign(delta/normalise)
      
        
