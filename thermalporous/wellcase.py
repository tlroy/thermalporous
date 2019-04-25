from firedrake import *

import thermalporous.utils as utils


class WellCase():
    
    def __init__(self, params, geo, well_case = None, prod_points = None, inj_points = None, constant_rate = False):
        self.name = 'Wells'
        self.Length = geo.Length
        self.Length_y = geo.Length_y
        if geo.dim == 3: 
            self.Length_z = geo.Length_z
        self.V = geo.V
        self.mesh = geo.mesh
        self.geo = geo
        self.params = params
        if constant_rate == True:
            self.flow_rate = self.flow_rate_constant
            self.flow_rate_twophase = self.flow_rate_twophase_constant
        else:
            self.flow_rate = self.flow_rate_
            self.flow_rate_twophase = self.flow_rate_twophase_
            
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
            elif well_case == "test0":
                Length = self.Length
                Length_y = self.Length_y
                prod_points = [[2., Length_y/4.], [2., Length_y/2.], [2., 3.*Length_y/4]] 
                inj_points = [[Length-2., Length_y/4.], [Length-2., Length_y/2.], [Length-2., 3.*Length_y/4]]
            elif well_case == "test":
                Lx = self.Length
                Ly = self.Length_y
                prod_points = [[10., 0.1*Ly], [10., 0.2*Ly], [10., 0.3*Ly], [10., 0.4*Ly], [10., 0.5*Ly], [10., 0.6*Ly], [10., 0.7*Ly], [10., 0.8*Ly], [10., 0.9*Ly]]
                inj_points = [[Lx - 10., 0.1*Ly], [Lx - 10., 0.2*Ly], [Lx - 10., 0.3*Ly], [Lx - 10., 0.4*Ly], [Lx - 10., 0.5*Ly], [Lx - 10., 0.6*Ly], [Lx - 10., 0.7*Ly], [Lx - 10., 0.8*Ly], [Lx - 10., 0.9*Ly]]
            elif well_case == "SPE10_40x40":
                Length = self.Length
                Length_y = self.Length_y
                Dx = self.geo.Dx
                prod_points = [[2*Dx, Length_y/4.], [2*Dx, Length_y/2.], [2*Dx, 3.*Length_y/4]] 
                inj_points = [[Length-2*Dx, Length_y/4.], [Length-2*Dx, Length_y/2.], [Length-2*Dx, 3.*Length_y/4]]
                
        elif self.geo.dim == 3:
            if well_case == "default":
                prod_points = [[self.Length/2, self.Length_y/2, self.Length_z*0.2]]
                inj_points = [[self.Length/2, self.Length_y/2, self.Length_z*0.8]]
            if well_case == "large":
                Lx = self.Length
                Ly = self.Length_y
                Lz = self.Length_z
                Dz = self.geo.Dz
                prod_points = [[Lx/8, Ly/2, Lz*0.2], [Lx/4, Ly/2, Lz*0.2], [3*Lx/8, Ly/2, Lz*0.2], [Lx/2, Ly/2, Lz*0.2], [5*Lx/8, Ly/2, Lz*0.2], [3*Lx/4, Ly/2, Lz*0.2], [7*Lx/8, Ly/2, Lz*0.2]] + [[Lx/8, Ly/4, Lz*0.2], [Lx/4, Ly/4, Lz*0.2], [3*Lx/8, Ly/4, Lz*0.2], [Lx/2, Ly/4, Lz*0.2], [5*Lx/8, Ly/4, Lz*0.2], [3*Lx/4, Ly/4, Lz*0.2], [7*Lx/8, Ly/4, Lz*0.2]] + [[Lx/8, 3*Ly/4, Lz*0.2], [Lx/4, 3*Ly/4, Lz*0.2], [3*Lx/8, 3*Ly/4, Lz*0.2], [Lx/2, 3*Ly/4, Lz*0.2], [5*Lx/8, 3*Ly/4, Lz*0.2], [3*Lx/4, 3*Ly/4, Lz*0.2], [7*Lx/8, Ly/4, Lz*0.2]]
                inj_points = [[Lx/8, Ly/2, Lz*0.8], [Lx/4, Ly/2, Lz*0.8], [3*Lx/8, Ly/2, Lz*0.8], [Lx/2, Ly/2, Lz*0.8], [5*Lx/8, Ly/2, Lz*0.8], [3*Lx/4, Ly/2, Lz*0.8], [7*Lx/8, Ly/2, Lz*0.8]] + [[Lx/8, Ly/4, Lz*0.8], [Lx/4, Ly/4, Lz*0.8], [3*Lx/8, Ly/4, Lz*0.8], [Lx/2, Ly/4, Lz*0.8], [5*Lx/8, Ly/4, Lz*0.8], [3*Lx/4, Ly/4, Lz*0.8], [7*Lx/8, Ly/4, Lz*0.8]] + [[Lx/8, 3*Ly/4, Lz*0.8], [Lx/4, 3*Ly/4, Lz*0.8], [3*Lx/8, 3*Ly/4, Lz*0.8], [Lx/2, 3*Ly/4, Lz*0.8], [5*Lx/8, 3*Ly/4, Lz*0.8], [3*Lx/4, 3*Ly/4, Lz*0.8], [7*Lx/8, Ly/4, Lz*0.8]]
        
        self.init_wells(prod_points, inj_points, 'circle')
        
    def init_wells(self, prod_points, inj_points, wellfunc):
        self.prod_wells = []
        self.inj_wells = []
        self.prodcount = 0
        self.injcount = 0
        for point in prod_points:
            self.prod_wells.append(self.make_well(point, wellfunc, 'prod'))
        for point in inj_points:
            self.inj_wells.append(self.make_well(point, wellfunc, 'inj'))
        
    def make_well(self, w, wellfunc, welltype):
        rate = self.params.rate 
        p_inj = self.params.p_inj 
        p_prod = self.params.p_prod 
        #node = utils.GetNodeClosestToCoordinate(self.V, w)
        if wellfunc == 'delta':
            delta = self.well_delta(w)
        elif wellfunc == 'circle':
            if self.geo.dim == 2:
                delta = self.well_circle(w)
            elif self.geo.dim == 3:
                delta = self.well_circle3D(w)
        
        
        if welltype == 'prod':
            current_count = str(self.prodcount)
            bhp = p_prod
            max_rate = -rate
            self.prodcount += 1
        elif welltype == 'inj':
            current_count = str(self.injcount)
            bhp = p_inj
            max_rate = rate
            self.injcount +=1

        return {'name': welltype + current_count, 'bhp': bhp, 'location': w,
                #'node': node,
                'delta': delta, 'max_rate': max_rate, 'rate': 0.0}

    def well_circle(self, w):
        xw = w[0]
        yw = w[1]
        x, y = SpatialCoordinate(self.mesh)
        radius = 0.1 #0.1875*0.3048 # 0.1 radius of well # 0.1875*0.3048 from ChenZhang2009
        delta = Function(self.V)
        delta.assign(interpolate(conditional(pow(x-xw,2)+pow(y-yw,2)<pow(radius,2), exp(-(1.0/(-pow(x-xw,2)-pow(y-yw,2)+pow(radius,2)))), 0.0), self.V))
        normalise = assemble(delta*dx)
        if normalise == 0:  
            #print("Using delta")
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
        height = 1.0 # WARNING: if using SPE10, this needs to be smaller if you one single cell wells. Or can just use 'delta' or SourceTerms instead
        delta = Function(self.V)
        delta.assign(interpolate(conditional(pow(x-xw,2)+pow(y-yw,2)<pow(radius,2), conditional(abs(z-zw)<height, 1.0, 0.0)*exp(-(1.0/(-pow(x-xw,2)-pow(y-yw,2)+pow(radius,2)))), 0.0), self.V))
        normalise = assemble(delta*dx)
        if normalise == 0:  
            #print("Using delta")
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
        #normalise = assemble(delta*dx)
        if self.geo.dim == 2:
            normalise = self.geo.Dx*self.geo.Dy
        elif self.geo.dim == 3:
            normalise = self.geo.Dx*self.geo.Dy*self.geo.Dz
        return delta.assign(delta/normalise)
    
    def flow_rate_(self, p, T, well, phase = 'oil'):
        #volumetric flow rate of wells using Peaceman model
        #Ignoring gravity for now
        import math
        bhp = well['bhp']
        max_rate = well['max_rate']
        if phase == 'oil':
            mu = self.params.oil_mu(T)
        elif phase == 'water':
            mu = self.params.water_mu(T)
        K_x = self.geo.K_x
        K_y = self.geo.K_y
        h = 5.0 # height of well opening #100*0.3048 from ChenZhang2009
        rw = 0.1 # 0.1 radius of well # 0.1875*0.3048 from ChenZhang2009
        #Dx = self.geo.Dx # if well is only one cell
        #Dy = self.geo.Dy
        Dx = 5.0 # length of triangles in ChenZhang2009: 300ft
        Dy = 5.0 # 5
        #ro = 0.14*(self.geo.Dx**2 + self.geo.Dy**2)**0.5 # equivalent radius
        ro = 0.28*((K_y/K_x)**0.5 * Dx**2 + (K_x/K_y)**0.5 * Dy**2)**0.5 / ( (K_y/K_x)**0.25 + (K_x/K_y)**0.25)
        #ro = 2.5
        Ke = (K_x*K_y)**0.5 #effective permeability
        is_prod = max_rate<0.0
        factor = 2*pi*h*Ke/ln(ro/rw)/mu
        if is_prod:
            dd = conditional(ge(bhp - p, 0.0), 0.0, (bhp - p))
        else:
            dd = conditional(le(bhp - p, 0.0), 0.0, (bhp - p))
        rate = factor*dd
        rate = conditional(ge(abs(rate) - abs(max_rate), 0.0), max_rate, rate) 
        return rate
    
    def flow_rate_constant(self, p, T, well, phase = 'oil'):
        return well['max_rate']
    
    def flow_rate_twophase_(self, p, T, well, S_o = 1.0):
        #volumetric flow rate of wells using Peaceman model
        import math
        bhp = well['bhp']
        max_rate = well['max_rate']
        #print(assemble(well['delta']*T*dx))
        oil_mu = self.params.oil_mu
        water_mu = self.params.water_mu
        mu = 1.0/(S_o/oil_mu(T) + (1.0-S_o)/water_mu(T))
        
        K_x = self.geo.K_x
        K_y = self.geo.K_y
        h = 5.0 # height of well opening #100*0.3048 from ChenZhang2009
        rw = 0.1 # 0.1 radius of well # 0.1875*0.3048 from ChenZhang2009
        #Dx = self.geo.Dx # if well is only one cell
        #Dy = self.geo.Dy
        Dx = 5.0 # length of triangles in ChenZhang2009: 300ft
        Dy = 5.0 # 5
        #ro = 0.14*(self.geo.Dx**2 + self.geo.Dy**2)**0.5 # equivalent radius
        ro = 0.28*((K_y/K_x)**0.5 * Dx**2 + (K_x/K_y)**0.5 * Dy**2)**0.5 / ( (K_y/K_x)**0.25 + (K_x/K_y)**0.25)
        #ro = 2.5
        Ke = (K_x*K_y)**0.5 #effective permeability
        is_prod = max_rate<0.0
        factor = 2*pi*h*Ke/ln(ro/rw)/mu
        if is_prod:
            dd = conditional(ge(bhp - p, 0.0), 0.0, (bhp - p))
        else:
            dd = conditional(le(bhp - p, 0.0), 0.0, (bhp - p))
        rate = factor*dd
        rate = conditional(ge(abs(rate) - abs(max_rate), 0.0), max_rate, rate) 
        water_rate = (1-S_o)/water_mu(T)*mu*rate
        oil_rate = S_o/oil_mu(T)*mu*rate
        return [rate, water_rate, oil_rate]
    
    def flow_rate_twophase_constant(self, p, T, well, S_o = 1.0):
                #volumetric flow rate of wells using Peaceman model
        import math
        bhp = well['bhp']
        max_rate = well['max_rate']
        #print(assemble(well['delta']*T*dx))
        oil_mu = self.params.oil_mu
        water_mu = self.params.water_mu
        mu = 1.0/(S_o/oil_mu(T) + (1.0-S_o)/water_mu(T))
        K_x = self.geo.K_x
        K_y = self.geo.K_y
        h = 5.0 # height of well opening #100*0.3048 from ChenZhang2009
        rw = 0.1 # 0.1 radius of well # 0.1875*0.3048 from ChenZhang2009
        #Dx = self.geo.Dx # if well is only one cell
        #Dy = self.geo.Dy
        Dx = 5.0 # length of triangles in ChenZhang2009: 300ft
        Dy = 5.0 # 5
        #ro = 0.14*(self.geo.Dx**2 + self.geo.Dy**2)**0.5 # equivalent radius
        ro = 0.28*((K_y/K_x)**0.5 * Dx**2 + (K_x/K_y)**0.5 * Dy**2)**0.5 / ( (K_y/K_x)**0.25 + (K_x/K_y)**0.25)
        #ro = 2.5
        Ke = (K_x*K_y)**0.5 #effective permeability
        is_prod = max_rate<0.0
        factor = 2*pi*h*Ke/ln(ro/rw)/mu
        if is_prod:
            dd = conditional(ge(bhp - p, 0.0), 0.0, (bhp - p))
        else:
            dd = conditional(le(bhp - p, 0.0), 0.0, (bhp - p))
        rate = max_rate
        water_rate = (1-S_o)/water_mu(T)*mu*rate
        oil_rate = S_o/oil_mu(T)*mu*rate
        return [rate, water_rate, oil_rate]
