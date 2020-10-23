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
    sample = EIT_testset[10]
    inputs, mask = sample["capacitance_data"], sample["segmentation_mask"]
    with torch.no_grad():
        inputs = inputs.unsqueeze(0)
        output = net(inputs)
        out_np = output.numpy()
        mask_np = mask.numpy()

    out_np = np.squeeze(out_np)
    mask_np = np.squeeze(mask_np)
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    ax1.imshow(out_np)
    ax2.imshow(mask_np)
    plt.show()