import torch
import numpy as np
from torch.utils.data import DataLoader
from net.UNet import Net
from dataHandler.dataParser import parse_data
from utilities import get_config
import matplotlib.pyplot as plt

if __name__ == '__main__':
    config = get_config()
    PATH, test_dir, batch_size = config["PATH"], config["train_dir"], config["batch_size"]
    EIT_testset = parse_data(root_dir=test_dir)
    net = Net()
    net.double()
    net.load_state_dict(torch.load(PATH))
    sample = EIT_testset[4260]
    inputs, mask = sample["capacitance_data"], sample["segmentation_mask"]
    sample_2 = EIT_testset[997]
    inputs_2, mask_2 = sample_2["capacitance_data"], sample_2["segmentation_mask"]
    with torch.no_grad():
        inputs = inputs.unsqueeze(0)
        output = net(inputs)
        out_np = output.numpy()
        mask_np = mask.numpy()
        inputs_2 = inputs_2.unsqueeze(0)
        output_2 = net(inputs_2)
        out_np_2 = output_2.numpy()
        mask_np_2 = mask_2.numpy()

    out_np = np.squeeze(out_np)
    mask_np = np.squeeze(mask_np)
    out_np_2 = np.squeeze(out_np_2)
    mask_np_2 = np.squeeze(mask_np_2)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
    ax1.imshow(out_np)
    ax2.imshow(mask_np)
    ax3.imshow(out_np_2)
    ax4.imshow(mask_np_2)
    plt.show()