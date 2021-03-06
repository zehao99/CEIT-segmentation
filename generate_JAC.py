import numpy as np
from MyEIT.readmesh import init_mesh
from MyEIT.readmesh import read_mesh_from_csv
from MyEIT.ejac import EJAC


init_mesh(True)

read_mesh = read_mesh_from_csv()
mesh_obj, _, _, _ = read_mesh.return_mesh()

jac_calc = EJAC(mesh_obj)
jac_calc.JAC_calculation()
jac_calc.show_JAC()
