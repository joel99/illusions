import pathlib
import os
import os.path as osp
from PIL import Image

from yacs.config import CfgNode as CN
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

class UniformityDataset(Dataset):
    def __init__(
        self,
        config,
        split='train',
        dataset_root='./data/UNIFORMITY'
    ):
        super().__init__()
        dataset_root = osp.join(dataset_root, split)
        self.all_paths = list(map(lambda x: osp.join(dataset_root, x), os.listdir(dataset_root)))

    @staticmethod
    def preprocess(img):
        img = F.resize(img, (128, 128))
        # * Note, uniformity images are encoded 0-1, ensure this is true in other datasets
        return F.to_tensor(img) - 0.5 # 0-center.

    def __len__(self):
        return len(self.all_paths)

    def __getitem__(self, index):
        img = Image.open(self.all_paths[index])
        return UniformityDataset.preprocess(img)