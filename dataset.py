import pathlib
import os.path as osp
from yacs.config import CfgNode as CN
import torch
from torch.utils.data import Dataset

class IllusionsDataset(Dataset):
    def __init__(
        self,
        config, # skipped for now
        split="train",
        dataset_root="./data/uniformity/",
    ):
        super().__init__()
        self.images = torch.load(osp.join(dataset_root, f"{split}.pth"))

    def __len__(self):
        r""" Number of samples. """
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index]
