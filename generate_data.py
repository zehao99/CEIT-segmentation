import matplotlib.pyplot as plt
from MyEIT.efem import EFEM
from MyEIT.ejac import EJAC
from MyEIT.readmesh import read_mesh_from_csv
from MyEIT.solver import Solver
import numpy as np
import csv
import progressbar

def normalize_data(data):
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    return data


if __name__ == "__main__":
    """ Read mesh from csv files(after initialization) """
    read_mesh = read_mesh_from_csv()
    mesh_obj, electrode_num, electrode_centers, radius = read_mesh.return_mesh()

    """Open Data File"""
    data_file = open("./data/generated_data.csv", "a", newline='')
    segmentation_file = open("./data/segmentation_mask.csv", "a", newline='')
    data_writer = csv.writer(data_file, delimiter=',')
    segmentation_writer = csv.writer(segmentation_file, delimiter=',')

    """Create EJAC and Solver Element"""
    fwd = EJAC(mesh_obj)
    solver = Solver()
    original_potential = np.copy(fwd.electrode_original_potential)

    """Generate Random change"""
    for i in progressbar.progressbar(range(10000)):
        sample_num = int(np.floor(np.random.rand() * 3) + 1)
        r = np.random.rand(sample_num) * 9 * 0.002 + 0.002
        obj_x = np.random.rand(sample_num) * (0.050 - 2 * r) - (0.025 - r)
        obj_y = np.random.rand(sample_num) * (0.050 - 2 * r) - (0.025 - r)
        val = np.power(10, np.random.rand(sample_num) * 2.3) * 5e-9
        shapes = []
        for j in range(sample_num):
            shape = "square" if np.random.rand() > 1 / 2 else "circle"
            shapes.append(shape)
        """Change capacitance"""
        for j in range(sample_num):
            fwd.fwd_FEM.change_add_capa_geometry([obj_x[j], obj_y[j]], r[j], val[j], shapes[j])

        """Generate GT and data"""
        segmentation = fwd.fwd_FEM.elem_capacitance
        segmentation = fwd.delete_outside_detect(segmentation)
        fwd.calc_origin_potential()
        changed_potential = np.copy(fwd.electrode_original_potential)
        generated_graph = solver.solve(changed_potential - original_potential)

        """Change back capacitance"""
        for j in range(sample_num):
            fwd.fwd_FEM.change_add_capa_geometry([obj_x[j], obj_y[j]], r[j], -val[j], shapes[j])

        """Write data"""
        data_writer.writerow(generated_graph)
        segmentation_writer.writerow(segmentation)

    """Close Data files"""
    data_file.close()
    segmentation_file.close()

