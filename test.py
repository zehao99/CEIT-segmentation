import numpy as np
import csv
import matplotlib.pyplot as plt
from dataHandler.capaToMatrix import CapaToMatrix
from MyEIT.utilities import read_csv_one_line_from_file

data1 = read_csv_one_line_from_file(filename="segmentation_mask.csv", path_name="./data", idx=10)
data2 = read_csv_one_line_from_file(filename="segmentation_mask.csv", path_name="./data", idx=11)
converter = CapaToMatrix()
data = np.array([data1, data2])
pic = converter.multi_row_transfer(data)

print(pic.shape)

