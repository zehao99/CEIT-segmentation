import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import evalutaion
from dataHandler.dataParser import parse_data
from net.UNet import Net
from utilities import get_config

writer = SummaryWriter('training_log/EIT_segmentation_1')


def matplot_img(output, mask):
    out_np = output.detach().numpy()
    mask_np = mask.numpy()
    out_np = np.squeeze(out_np)
    predict = out_np > 0.5
    mask_np = np.squeeze(mask_np)
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
    ax1.imshow(out_np)
    ax2.imshow(predict)
    ax3.imshow(mask_np)
    return fig


def weights_init(m):
    if isinstance(m, nn.Conv1d):
        nn.init.normal_(m.weight, mean=0.01, std=0.01)
        nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.001, std=0.001)
        nn.init.constant_(m.bias, 0)


class My_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.mean(torch.pow(torch.abs(x - y) * 100, 2))


if __name__ == '__main__':
    # load config
    config = get_config()
    PATH, epoch_num, batch_size, learning_rate, root_dir = \
        config["PATH"], config["epoch"], config["batch_size"], config["learning_rate"], config["train_dir"]
    # load dataset
    EIT_dataset = parse_data(root_dir=root_dir)
    # get the testing data
    sample = EIT_dataset[2487]
    test_inputs, test_mask = sample["capacitance_data"], sample["segmentation_mask"]
    test_inputs = test_inputs.unsqueeze(0)
    sample_2 = EIT_dataset[1024]
    test_inputs_2, test_mask_2 = sample_2["capacitance_data"], sample_2["segmentation_mask"]
    test_inputs_2 = test_inputs_2.unsqueeze(0)
    trainloader = DataLoader(EIT_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    # set device
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    # determine nn
    net = Net()
    net.apply(weights_init)
    # net.load_state_dict(torch.load(PATH))
    criterion = torch.nn.BCELoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    minibatch_print = 100
    net.to(device)
    net.double()
    # start training
    item_num = 0
    max_F1_record = 0
    for epoch in range(epoch_num):
        print("Training epoch %d" % epoch)
        running_loss = 0.0
        sum_F1 = 0
        item_num = 0
        for i, data in enumerate(trainloader, 0):
            inputs, masks = data["capacitance_data"].to(device), data["segmentation_mask"].to(device)
            item_num += inputs.size()[0]

            optimizer.zero_grad()
            out = net(inputs)
            SR_probs = torch.sigmoid(out)
            SR_flat = SR_probs.view(SR_probs.size(0), -1)

            GT_flat = masks.view(masks.size(0), -1)
            loss = criterion(SR_flat, GT_flat)
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            running_loss += loss.item()
            accuracy = evalutaion.get_accuracy(out, masks)
            F1_score = evalutaion.get_F1(out, masks)
            sum_F1 += F1_score

            test_out = net(test_inputs)
            img_1 = matplot_img(test_out, test_mask)
            writer.add_figure('Results on the model 1', img_1, global_step=epoch * len(trainloader) + i)
            test_out_2 = net(test_inputs_2)
            img_2 = matplot_img(test_out_2, test_mask_2)
            writer.add_figure('Results on the model 2', img_2, global_step=epoch * len(trainloader) + i)
            writer.add_scalar('training loss', batch_loss / batch_size, epoch * len(trainloader) + i)
            writer.add_scalar('Accuracy on trainset(0.5 threshold)', accuracy, epoch * len(trainloader) + i)
            writer.add_scalar('F1 score on trainset(0.5 threshold)', F1_score, epoch * len(trainloader) + i)
            # out_label = labels.cpu()
        # print(item_num)
        if sum_F1 / item_num >= max_F1_record:
            torch.save(net.state_dict(), PATH)
            max_F1_record = sum_F1 / item_num
        if epoch in [20, 40, 80, 160, 320]:
            learning_rate = 0.75 * learning_rate

        print('epoch %d,  loss: %.10f' % (epoch + 1, running_loss / item_num))

    print('Finished training.')
    # save model
    # torch.save(net.state_dict(), PATH)
