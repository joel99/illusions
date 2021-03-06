import pathlib
import os
import os.path as osp
from os.path import isfile
from PIL import Image

from yacs.config import CfgNode as CN
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import torchvision.transforms as transforms

class UniformityDataset(Dataset):
    def __init__(
        self,
        config,
        split='train',
        dataset_root='./data/UNIFORMITY',
        augment=[]
    ):
        super().__init__()
        dataset_root = osp.join(dataset_root, split)
        self.all_paths = list(map(lambda x: osp.join(dataset_root, x), os.listdir(dataset_root)))
        self.all_paths = [f for f in self.all_paths if isfile(f)]
        self.grayscale = config.TASK.CHANNELS == 1
        self.cache = {}
        self.transform = None
        if 'rotate' in augment and split == 'train':
            self.transform = transforms.RandomRotation(
                40,
                fill=(1,)
                # torch.arange(-40, 45, 5)
            )

    def preprocess(self, img):
        img = F.resize(img, (64, 64))
        if self.grayscale:
            img = F.to_grayscale(img)
        # * Note, uniformity images are encoded 0-1, ensure this is true in other datasets
        if self.transform is not None:
            img = self.transform(img)
        img = F.to_tensor(img)
        return img - 0.5

    @staticmethod
    def unpreprocess(view):
        # For viz
        # * Note, uniformity images are encoded 0-1, ensure this is true in other datasets
        return torch.clamp(view + 0.5, 0, 1) # 0 - 1

    def __len__(self):
        return len(self.all_paths)

    def __getitem__(self, index):
        if len(self) < 10 and index in self.cache:
            return self.cache[index]
        img = Image.open(self.all_paths[index])
        item = self.preprocess(img)
        if self.transform is not None and len(self) < 10:
            self.cache[index] = item
        return item
