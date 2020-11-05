import matplotlib.pyplot as plt
import numpy as np
from MyEIT.utilities import read_csv_one_line_from_file
from dataHandler.capaToMatrix import CapaToMatrix
trans = CapaToMatrix()

idx_1 = 1233

data1 = read_csv_one_line_from_file("generated_data_train.csv", "./data", idx_1)
data1 = trans.transfer_data_to_matrix(data1)

mask1 = read_csv_one_line_from_file("segmentation_mask_train.csv", "./data", idx_1)
mask1 = trans.transfer_data_to_matrix(mask1)

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

ax1.imshow(data1)
ax2.imshow(mask1)
plt.show()