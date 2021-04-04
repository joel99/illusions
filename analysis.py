#%%
from pathlib import Path
import os.path as osp
from yacs.config import CfgNode as CN

import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn.functional as F
from config.default import get_config
from model import SaccadingRNN
from dataset import UniformityDataset
# We need to grab the checkpoint, and its corresponding config.
# TODO figure out how to store the config in checkpoint and just load checkpoints
config = './config/base.yaml'
# config = './config/debug.yaml'
config = './config/debug2.yaml'
config = './config/base.yaml'

seed = 0
version = 10
version = 12
version = 14
version = 16
version = 18
version = 21
version = 40
version = 42
version = 43
version = 48
version = 49
version = 50
version = 57
version = 0
version = 58
version = 59
version = 63
version = 70
variant = osp.split(config)[1].split('.')[0]
config = get_config(config)

root = Path(f'runs/{variant}-{seed}/lightning_logs/')
# # * This is the default output, if you want to play around with a different checkpoint load it here.
model_ckpt = list(root.joinpath(f"version_{version}").joinpath('checkpoints').glob("*"))[0]

weights = torch.load(model_ckpt, map_location='cpu')
model = SaccadingRNN(config)
model.load_state_dict(weights['state_dict'])
model.eval()
dataset = UniformityDataset(config, split="train")

#%%
image = dataset[0]
image = dataset[1]
# image = dataset[2]
image = dataset[3]
print(dataset.all_paths[0])
print(image.size())
proc_view = UniformityDataset.unpreprocess(image).permute(1, 2, 0)
proc_view = proc_view.squeeze(-1)
plt.imshow(proc_view)

all_views, noised_views, patches, state = model.predict(image)

# Hm, doesn't seem to matter.... am I looking at the right output?
# Why is my loss higher than reported?

loss1 = F.mse_loss(all_views[0], patches[0])
loss1 = F.mse_loss(all_views[1], patches[1])
# loss1 = F.mse_loss(all_views[1], patches[0])
# loss1 = F.mse_loss(all_views[1:], patches)
print(loss1)
# loss2 = F.mse_loss(views, patches)
# F.mse_loss(views[1:], patches)

#%%
# It don't even look like the right image.
times = [0, 1, 5, 6]
f, axes = plt.subplots(len(times), 2, sharex=True, sharey=True)
for i, t in enumerate(times):
    true_image = noised_views[t, 0]
    # true_image = all_views[t, 0]
    # true_image = all_views[t + 1, 0]
    proc_true = UniformityDataset.unpreprocess(true_image).permute(1, 2, 0)
    proc_true = proc_true.squeeze(-1)

    # axes[i, 0].imshow(proc_true[..., 2])
    axes[i, 0].imshow(proc_true)
    pred_image = patches[t, 0]
    proc_pred = UniformityDataset.unpreprocess(pred_image).permute(1, 2, 0)
    proc_pred = proc_pred.squeeze(-1)
    # axes[i, 1].imshow(proc_pred[..., 2])
    axes[i, 1].imshow(proc_pred)
axes[0, 0].set_title('True')
axes[0, 1].set_title('Pred')
