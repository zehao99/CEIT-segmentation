from __future__ import print_function, division
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from .capaToMatrix import CapaToMatrix


class EITSegmentationDataset(Dataset):
    """
    Segmentation Dataset parser
    """

    def __init__(self, root_dir='./data/', transform=None):
        """
        Prepare dataset for segmentation

        Args:
            root_dir: data storage directory
            transform: transformer of data
        """
        self.capacitance_data_frame = pd.read_csv(root_dir + 'generated_data.csv', header=None)
        self.capacitance_data_frame = self.capacitance_data_frame[:]
        self.segmentation_data_frame = pd.read_csv(root_dir + 'segmentation_mask.csv', header=None)
        self.segmentation_data_frame = self.segmentation_data_frame[:]
        self.root_dir = root_dir
        self.transform = transform
        self.mat_converter = CapaToMatrix()

    def __len__(self):
        return len(self.capacitance_data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        capacitance_data = self.capacitance_data_frame.iloc[idx, :]
        capacitance_data = self.mat_converter.multi_row_transfer(capacitance_data)
        segmentation_mask = self.segmentation_data_frame.iloc[idx, :]
        segmentation_mask = self.mat_converter.multi_row_transfer(segmentation_mask)
        sample = {"capacitance_data": capacitance_data, "segmentation_mask": segmentation_mask}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """
    Convert ndArrays in sample to Tensors.
    """

    def __call__(self, sample):
        capacitance_data, segmentation_mask = sample["capacitance_data"], sample["segmentation_mask"]

        return {"capacitance_data": torch.from_numpy(capacitance_data).double(),
                "segmentation_mask": torch.from_numpy(segmentation_mask).double()}


def parse_data(root_dir='learning_data/'):
    eit_dataset = EITSegmentationDataset(root_dir=root_dir, transform=transforms.Compose([ToTensor()]))
    return eit_dataset


if __name__ == "__main__":
    parse_data()
