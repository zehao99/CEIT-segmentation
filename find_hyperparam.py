import numpy as np
import pandas as pd
from MyEIT.readmesh import read_mesh_from_csv
from MyEIT.efem import EFEM
from MyEIT.ejac import EJAC
import progressbar
import csv
import matplotlib.pyplot as plt


def find_hyperparam():
    # import mesh
    read_mesh = read_mesh_from_csv()
    mesh_obj, electrode_num, electrode_centers, radius = read_mesh.return_mesh()
    # extract node, element, alpha
    points = mesh_obj['node']
    tri = mesh_obj['element']
    x, y = points[:, 0], points[:, 1]
    test_center = [-10, -20]
    test_center_list = [[0, 0], [10, 10], [10, 20], [20, 10], [20, 20]]
    curr_mode = 'n'
    test_capa = 1e-12
    """ 1. problem setup """
    fwd = EJAC(mesh_obj)
    fwd.read_JAC_np()
    experiment = EJAC(mesh_obj)
    origin_capacitance = np.copy(experiment.initial_capacitance)

    """ 2. simulation setup """
    elem_param = np.copy(fwd.fwd_FEM.elem_param)
    object_tri = fwd.delete_outside_detect(tri)
    object_param = fwd.delete_outside_detect(elem_param)
    x_elem = object_param[:, 7]
    y_elem = object_param[:, 8]
    x_elem = np.reshape(x_elem, (-1, 1))
    y_elem = np.reshape(y_elem, (-1, 1))
    x_label = np.linspace(-40, 45, 250)
    x_2 = []
    y_2 = []
    # slice_list=[]
    # for i in [0,2,4,6,8,10,12,14]:
    #    for j in [0,2,4,6,8,10,12,14]:
    #        if j == i:
    #            pass
    #        if j < i:
    #            slice_list.append(i * 15 + j)
    #        if j > i:
    #            slice_list.append(i * 15 + j -1)

    for test_center in test_center_list:
        x_1 = []
        y_1 = []
        experiment.fwd_FEM.change_add_capa_geometry(test_center, 10, test_capa, "circle")
        experiment.calc_origin_potential()
        elem_capacitance_origin = experiment.fwd_FEM.elem_capacitance - origin_capacitance
        elem_capacitance_origin = np.reshape(elem_capacitance_origin, (-1, 1))
        elem_capacitance_origin = fwd.delete_outside_detect(elem_capacitance_origin)
        x_o = np.sum(elem_capacitance_origin * (x_elem) / np.max(elem_capacitance_origin))
        y_o = np.sum(elem_capacitance_origin * (y_elem) / np.max(elem_capacitance_origin))
        for i in progressbar.progressbar(x_label):
            electrode_potential = experiment.electrode_original_potential
            lmbda = 10 ** (i / 15)
            J = fwd.eliminate_non_detect_JAC() - 1
            R = np.eye(J.shape[1])
            delta_V = electrode_potential - np.copy(fwd.electrode_original_potential)
            # J = J[slice_list,:]#4 electrodes
            # delta_V = delta_V[slice_list]#4 electrodes
            c_predict = fwd.Msolve_gpu(J, R, lmbda, delta_V)
            c_predict = np.reshape(c_predict, (-1, 1))
            x_1.append(np.sum(c_predict * (x_elem) / np.max(c_predict)) - x_o)
            y_1.append(np.sum(c_predict * (y_elem) / np.max(c_predict)) - y_o)
        x_2.append(x_1)
        y_2.append(y_1)
        experiment.fwd_FEM.change_add_capa_geometry(test_center, 10, -test_capa, "circle")
    x_1 = np.sum(np.array(x_2), axis=0)
    y_1 = np.sum(np.array(y_2), axis=0)
    x_label = 10 ** (x_label / 15)
    data = np.array([x_label, x_1, y_1])
    df = pd.DataFrame(data.T)
    df.to_csv('./hyper_choose_center.csv')
    y_0 = np.zeros((250))
    plt.rc('font', family='serif', serif='Times')
    # plt.rc('text', usetex=True)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('axes', labelsize=14)
    fig, ax = plt.subplots(figsize=(4, 3.2))
    plt.subplots_adjust(left=0.13, right=0.96, top=0.96, bottom=0.11)
    ax.plot(x_label, x_1, label='x')
    ax.plot(x_label, y_1, label='y')
    # ax.plot(x_label, y_0, ls='--', label='zero')
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\Delta_{WCC}$")
    ax.legend()
    plt.grid(True)
    plt.show()


def draw_L_Curve():
    # import mesh
    read_mesh = read_mesh_from_csv()
    mesh_obj, electrode_num, electrode_centers, radius = read_mesh.return_mesh()
    # extract node, element, alpha
    points = mesh_obj['node']
    tri = mesh_obj['element']
    x, y = points[:, 0], points[:, 1]
    test_center = [-10, -20]
    curr_mode = 'n'
    test_capa = 1e-11
    """ 1. problem setup """
    fwd = EJAC(mesh_obj)
    fwd.read_JAC_np()
    experiment = EJAC(mesh_obj)
    origin_capacitance = experiment.initial_capacitance

    """ 2. simulation setup """
    x_label = np.linspace(-40, 45, 86)
    x_1 = []
    y_1 = []
    experiment.fwd_FEM.change_add_capa_geometry(test_center, 10, test_capa, "circle")
    experiment.calc_origin_potential()
    elem_capacitance_origin = experiment.fwd_FEM.elem_capacitance - origin_capacitance
    for i in progressbar.progressbar(x_label):
        electrode_potential = experiment.electrode_original_potential
        lmbda = 10 ** (i / 15)
        J = fwd.eliminate_non_detect_JAC() - 1
        R = np.eye(J.shape[1])
        delta_V = electrode_potential - np.copy(fwd.electrode_original_potential)
        c_predict = fwd.Msolve_gpu(J, R, lmbda, delta_V)
        y_1.append(np.log10(np.linalg.norm(np.dot(R, c_predict))))
        x_1.append(np.log10(np.linalg.norm(np.dot(J, c_predict) - delta_V)))
    data = np.array([x_1, y_1])
    df = pd.DataFrame(data)
    df.to_csv('./hyper_choose.csv')
    fig, ax = plt.subplots()
    ax.plot(x_1, y_1)
    offset = 0.04
    ax.annotate(r"$\lambda=51$", (-0.523907, -1.482682), xytext=(offset, offset), arrowprops=dict(arrowstyle="->"))
    ax.set_xlabel(r"$log_{10}||H\hat{\delta\sigma}||$")
    ax.set_ylabel(r"$log_{10}||R\hat{\delta\sigma}||$")
    plt.show()


if __name__ == "__main__":
    find_hyperparam()
