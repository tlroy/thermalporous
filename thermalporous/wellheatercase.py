from firedrake import *

from thermalporous.wellcase import WellCase
from thermalporous.heatercase import HeaterCase

class WellHeaterCase(WellCase, HeaterCase):
    
    def __init__(self, params, geo, well_case = None, prod_points = list(), inj_points = list(), constant_rate = False):
        
        WellCase.__init__(self, params, geo, well_case = well_case, prod_points = prod_points, inj_points = inj_points, constant_rate = constant_rate)
        HeaterCase.__init__(self, params, geo, well_case = well_case, heater_points = inj_points + prod_points)
        self.name = 'Wells and Heaters'
