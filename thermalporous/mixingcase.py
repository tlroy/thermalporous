from firedrake import *

class MixingCase():

    def __init__(self, params, geo, mixing_case = "coldandhot"):
        self.name = "Mixing"
        self.V = geo.V
        self.geo = geo
        self.params = params
        self.mixing_case = mixing_case
        if not (mixing_case in ["coldonhot", "coldandhot", "heavyonlight"]):
            raise SystemExit("Error: Undefined mixing_case: %s" % mixing_case)
        if (mixing_case == "coldonhot" and geo.dim == 2):
            print("Warning: coldonhot only implemented for 3D cases. Switching to coldandhot")
            self.mixing_case = "coldandhot"
        if mixing_case == "heavyonlight":
            # Need to make oil lighter than default parameters
            self.params.API = 40

    def init_IC(self, phases = "Single phase"):
        W = self.geo.W # defined in thermal model
        ic = Function(W)
        x = SpatialCoordinate(self.V.mesh())
        T_hot = self.params.T_inj
        T_cold = self.params.T_prod
        p_ref = self.params.p_ref
        if self.mixing_case == "coldandhot":
            ic_T = interpolate(conditional(gt(x[1],self.geo.Length_y/2.), T_cold, T_hot), self.V)
            ic.sub(0).assign(Constant(p_ref))
            ic.sub(1).assign(ic_T)
        elif self.mixing_case == "coldonhot":
            ic_T = interpolate(conditional(gt(x[2],self.geo.Length_z/2.), T_cold, T_hot), self.V)
            ic.sub(0).assign(Constant(p_ref))
            ic.sub(1).assign(ic_T)
        elif self.mixing_case == "heavyonlight":
            if phases == "Single phase": raise SystemExit("Error: heavyonlight case not defined for Single Phase")
            ic_S = interpolate(conditional(gt(x[2],self.geo.Length_z/2.), 0.0, 1.0), self.V)
            ic.sub(0).assign(Constant(p_ref))
            ic.sub(1).assign(Constant(T_cold))
            ic.sub(2).assign(ic_S)
        if self.mixing_case in ["coldandhot", "coldonhot"] and phases == "Two-phase":
            S_o = self.params.S_o
            ic.sub(2).assign(Constant(S_o))
        return ic
