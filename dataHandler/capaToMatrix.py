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
        """
        Transfer mesh related data to a matrix image

        Assign the value to the image pixel and get the mean value on every pixel.

        Args:
            data: NdArray (mesh_num) the capacitance data.

        Returns:
            object: NdArray DIMENSION * DIMENSION matrix
        """
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
        """
        Transfer several lines of raw data into image arrays

        read every line and transfer it to matrix then reshape to fit with training input.

        Args:
            data: NdArray (len, mesh_num) the capacitance data.

        Returns:
            object:  NdArray (Sample_num, 1, DIMENSION, DIMENSION) size of data
        """
        out = np.zeros((len(data), 1, self.DIMENSION, self.DIMENSION))
        for i, row in enumerate(data):
            mat = self.transfer_data_to_matrix(row)
            out[i] = mat[np.newaxis, :, :]
        return out

    def single_row_transfer(self, data):
        """
        Transfer one lines of raw data into image arrays

        read one line and transfer it to matrix then reshape to fit with training input.

        Args:
            data: NdArray (mesh_num) the capacitance data.

        Returns:
            object: NdArray (1, DIMENSION, DIMENSION) size of data
        """
        out = np.zeros((self.DIMENSION, self.DIMENSION, 1))
        mat = self.transfer_data_to_matrix(data)
        out = mat[np.newaxis, :, :]
        return out

    def get_coordinate(self, x, y, pixel_size):
        """
        Transfer (x, y) coordinate in to row col inside the output matrix
        """
        row = int((x + self.DIMENSION * pixel_size / 2) / pixel_size)
        col = int((y + self.DIMENSION * pixel_size / 2) / pixel_size)
        return row, col

    def single_mask_transfer(self, data):
        """
        Mask Transformation transfer mask to 0 and 1

        Judge if the number at the position is bigger than 1e-9

        Args:
            data: NdArray (mesh_num) the mask data

        Returns:
            object: segmentation mask
        """
        mat = self.transfer_data_to_matrix(data)
        out = mat[np.newaxis, :, :] > 1e-9
        return out
