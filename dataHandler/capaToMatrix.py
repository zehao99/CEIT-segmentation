import numpy as np
from MyEIT.solver import Solver
from MyEIT.utilities import read_csv_from_file, get_config


class CapaToMatrix:
    def __init__(self):
        self.config = get_config()
        self.solver = Solver()
        self.elem_param = np.copy(self.solver.elem_param)
        self.elem_param = self.solver.delete_outside_detect(self.elem_param)
        self.DIMENSION = 40

    def transfer_data_to_matrix(self, data):
        bound_size = self.config["detection_bound"]
        pixel_size = bound_size * 2 / self.DIMENSION
        sum_mat = np.zeros((self.DIMENSION, self.DIMENSION))
        count_mat = np.zeros((self.DIMENSION, self.DIMENSION))
        for i, capa in enumerate(data):
            x = self.elem_param[i][7]
            y = self.elem_param[i][8]
            row, col = self.get_coordinate(x, y, pixel_size)
            sum_mat[row][col] += capa
            count_mat[row][col] += 1
        return sum_mat / count_mat

    def multi_row_transfer(self, data):
        out = np.zeros((len(data), self.DIMENSION, self.DIMENSION, 1))
        for i, row in enumerate(data):
            mat = self.transfer_data_to_matrix(row)
            out[i] = mat[np.newaxis, :, :]
        return out

    def single_row_transfer(self, data):
        out = np.zeros((self.DIMENSION, self.DIMENSION, 1))
        mat = self.transfer_data_to_matrix(data)
        out = mat[np.newaxis, :, :]
        return out

    def get_coordinate(self, x, y, pixel_size):
        row = int((x + self.DIMENSION * pixel_size / 2) / pixel_size)
        col = int((y + self.DIMENSION * pixel_size / 2) / pixel_size)
        return row, col

    def single_mask_transfer(self, data):
        mat = self.transfer_data_to_matrix(data)
        out = mat[np.newaxis, :, :] > 1e-9
        return out
