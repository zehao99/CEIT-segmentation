import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from net.UNet import Net
from dataHandler.dataParser import parse_data
from utilities import get_config


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
    trainloader = DataLoader(EIT_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    # set device
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    # determine nn
    net = Net()
    net.apply(weights_init)
    # net.load_state_dict(torch.load(PATH))
    criterion = torch.nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    minibatch_print = 100
    net.to(device)
    net.double()
    # start training
    item_num = 0
    for epoch in range(epoch_num):
        print("Training epoch %d" % epoch)
        running_loss = 0.0
        item_num = 0
        for i, data in enumerate(trainloader, 0):
            inputs, masks = data["capacitance_data"].to(device), data["segmentation_mask"].to(device)
            item_num += inputs.size()[0]
            optimizer.zero_grad()
            out = net(inputs)
            loss = criterion(out, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # out_label = labels.cpu()
        # print(item_num)
        print('epoch %d,  loss: %.10f' % (epoch + 1, running_loss / item_num))

    print('Finished training.')
    print(out.cpu(), masks.cpu())
    # save model
    torch.save(net.state_dict(), PATH)
