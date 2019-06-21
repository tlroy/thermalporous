from firedrake import *

class PhysicalParameters():
    # Define parameters in SI units
    #K = 3E-13 # Permeability in m^2.
    #mu = 1E-3 # viscosity at reference pressure in Pa*s. at 20C
    #phi = 0.2 # rock porosity
    
    ko = 0.15 # conductivity of oil in W/m*K/ at 20C
    kw = 0.6005638 # conductivity of water in W/m*K 
    #kr = 2.5 # conductivity of sandstone
    kr = 1.7295772056 # conductivity of rock from SPE4 W/m*K
    c_v_w = 4181.3 # specific heat of water in J/(K*k). at 25C
    c_v_o = 2093.4 # specific oil (SPE4 from Zhangxin) J/(K*kg)
    c_r = 920.0 # specific heat of sandstone in J/(K*kg)
    rho_r = 2650.0 # density of sandstone
    p_inj = 6.895e7*1e-6  #pressure at injection well 6.8948e7 Pa for SPE10; 
    p_prod = 2.7579e7*1e-6 #pressure at production well 2.7579e7 Pa for SPE10
    #T_inj = 505.372 # from SPE4 Zhangxin steam temperature 370.0
    T_inj = 422.039 # 300F. max oil_mu value
    #T_inj = 372.15 # 99C
    #T_prod = 324.817 #290.0 #temperature in the reservoir SPE4 Zhangxin
    T_prod = 288.706 #60F
    API = 10.0 #viscosity formula valid for 10 < API < 20, i.e. heavy oil_mu # gravity of oil at 60F, heavy oil: between 10 and 20
    p_ref = 4.1369e7*1e-6 # SPE10 reference pressure 4.1369e7 Pa
    T_ref = (T_inj+T_prod)/2.0 # reference temperature
    g = 9.80665*1e-6 #9.80665 m/s^2 acceleration due to gravity
    S_o = 1.0 # default initial oil saturation
    U = 5.44409e6 # heat coefficient for heaters J/(s*K)
    rate = 1.8e-3 # max inj/prod rate#-0.00184013 # SPE4 #-1e-7 #Flow rate m^3/s Peaceman: 3.2774e-7
    
    def oil_rho(self, p, T):
        # Formula for density, using coefficients representative of reservoir simulation
        SG = 141.5/(self.API + 131.5)
        rho_ref = SG*999.0
        c = 5.5e-5 # 5.5e-5 for paper compressibility in 1/bar
        p0 = 1.01325 # reference pressure in bar
        e1 = 2.5e-4 # 2.5e-4 for paper thermal expansivity coefficient in 1/K
        T0 = 15.5556 + 273.15
        pbar = p*1e1 #*1e-5
        return rho_ref*e**(c*(pbar-p0))*e**(-e1*(T-T0))
    
    def oil_mu(self, T):
        # Second Bennison's formula for dead oil in kg*m-1*s-1 from Bennison1998
        # viscosity formula valid for 10 < API < 20
        # only makes sense T < 300F
        A1 = -0.8021
        A2 = 23.8765
        A3 = 0.31458
        A4 = -9.21592
        Tf = 1.8*(T - 273.15) + 32.0 # temperature in Fahrenheit
        return 1E-3*( 10.0**(A1*self.API + A2) * Tf**(A3*self.API + A4) )
    
    
    def water_rho(self, p, T):
        # Trangenstein's modification of Kell's 1975 correlation
        E_0 = 999.83952
        E_1 = 16.955176
        E_2 = -7.987E-3
        E_3 = -46.170461E-6
        E_4 = 105.56302E-9 
        E_5 = -280.54353E-12
        E_6 = 16.87985E-3 
        E_7 = 10.2
        Cw = 3.98854E-4 # water compressibility in MPA^-1
        Tc = T - 272.15 # temperature in Celsius
        pp = p #*1e-6 pressure in Pa# pressure in MPA
        return (E_0 + E_1*Tc + E_2*Tc**2 + E_3*Tc**3 + E_4*Tc**4 + E_5*Tc**5)*e**(Cw*(pp-E_7))/(1 + E_6*Tc)
    
    def water_mu(self, T):
        # Grabowski's formula (Grabowski 1979)
        Aw = 2.1850
        Bw = 0.04012
        Cw = 5.1547E-6
        Tf = 1.8*(T - 272.15) + 32 # temperature in Fahrenheit
        return 1E-3*Aw/(-1 + Bw*Tf + Cw*Tf**2)
    
    def rel_perm_o(self, S_o):
        # relative permeability of oil
        return S_o
    
    def rel_perm_w(self, S_o):
        # relative permeability of water
        return 1.0 - S_o


