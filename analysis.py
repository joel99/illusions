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
config = './config/debug.yaml'
# config = './config/debug2.yaml'
# config = './config/base.yaml'
# config = './config/base_e2e.yaml'
config = './config/large.yaml'
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
version = 72
version = 73
version = 84
version = 85
version = 1
version = 5
version = 0

variant = osp.split(config)[1].split('.')[0]
config = get_config(config)

root = Path(f'runs/{variant}-{seed}/lightning_logs/')
# * This is the default output, if you want to play around with a different checkpoint load it here.
model_ckpt = list(root.joinpath(f"version_{version}").joinpath('checkpoints').glob("*"))[0]

weights = torch.load(model_ckpt, map_location='cpu')
model = SaccadingRNN(config)
model.load_state_dict(weights['state_dict'])
model.eval()
dataset = UniformityDataset(config, split="train")

#%%
index = 0
# index = 2
# index = 9
index = 25000

# index = 700
# index = 750
# index = 800
image = dataset[index]
proc_view = UniformityDataset.unpreprocess(image).permute(1, 2, 0)
proc_view = proc_view.squeeze(-1)
plt.imshow(proc_view)

saccades = model._generate_saccades(image)
all_views, noised_views, patches, state = model.predict_with_saccades(image, saccades, mode='predictive_patch')
# Note, simply using saccades twice in a row is OOD.

# all_views, noised_views, patches, state = model.predict(image)

# Hm, doesn't seem to matter.... am I looking at the right output?
# Why is my loss higher than reported?

print(all_views.size(), patches.size())
loss1 = F.mse_loss(all_views[1], patches[0])
print(loss1)
plt.imshow(image.squeeze(0))

#%%
# It don't even look like the right image.
times = [0, 10, 20, 30]
f, axes = plt.subplots(len(times), 2, sharex=True, sharey=True)
for i, t in enumerate(times):
    true_image = noised_views[t + all_views.size(0) - patches.size(0), 0]
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

plt.savefig('test.png')

#%%
import numpy as np
step = 3
grid_h = np.linspace(0, image.size(-2), step)
grid_w = np.linspace(0, image.size(-1), step)
grid_x, grid_y = np.meshgrid(grid_h, grid_w)
grid = torch.stack([torch.tensor(grid_x), torch.tensor(grid_y)], dim=-1).long()

grid = grid.flatten(0, 1)
step_state = state.expand(grid.size(0), -1, -1)
patches = model._predict_at_location(step_state, grid, mode='patch').detach()
print(patches.size())

# Assemble patches
w_span, h_span = model.cfg.FOV_WIDTH // 2, model.cfg.FOV_HEIGHT // 2
padded_image = F.pad(image.squeeze(0), (w_span, w_span, h_span, h_span))
belief = torch.zeros_like(padded_image).detach()
# Pad image
for patch, loc in zip(patches, grid):
    belief[
        loc[0]: loc[0] + 2 * w_span,
        loc[1]: loc[1] + 2 * h_span
    ] = patch
#%%
f, axes = plt.subplots(1, 2, sharex=True, sharey=True)
axes[0].scatter(*(saccades.T + w_span), color='white')
axes[0].imshow(padded_image)
axes[0].set_title('True')
axes[0].axis('off')
axes[1].imshow(belief)
axes[1].set_title('Perceived')
axes[1].axis('off')
plt.savefig('test.png', dpi=300)

#%%
plt.imshow(image.squeeze(0))
plt.savefig('test.png', dpi=300)

#%%
all_views, noised_views, patches, state = model.predict_with_saccades(image, saccades, mode='predictive_patch')
fixate_saccades = model._generate_saccades(image, mode='fixate')
all_views, noised_views, patches, fixate_state = model.predict_with_saccades(image, fixate_saccades, mode='predictive_patch', initial_state=None)
# all_views, noised_views, patches, fixate_state = model.predict_with_saccades(image, fixate_saccades, mode='predictive_patch', initial_state=state)

step = 3
grid_h = np.linspace(0, image.size(-2), step)
grid_w = np.linspace(0, image.size(-1), step)
grid_x, grid_y = np.meshgrid(grid_h, grid_w)
grid = torch.stack([torch.tensor(grid_x), torch.tensor(grid_y)], dim=-1).long()

grid = grid.flatten(0, 1)
step_state = state.expand(grid.size(0), -1, -1)
step_fixate = fixate_state.expand(grid.size(0), -1, -1)
patches = model._predict_at_location(step_state, grid, mode='patch').detach()
fixate_patches = model._predict_at_location(step_fixate, grid, mode='patch').detach()

# Assemble patches
w_span, h_span = model.cfg.FOV_WIDTH // 2, model.cfg.FOV_HEIGHT // 2
padded_image = F.pad(image.squeeze(0), (w_span, w_span, h_span, h_span))
belief = torch.zeros_like(padded_image).detach()
fixate_belief = torch.zeros_like(padded_image).detach()
# Pad image
for patch, fixate_patch, loc in zip(patches, fixate_patches, grid):
    belief[
        loc[0]: loc[0] + 2 * w_span,
        loc[1]: loc[1] + 2 * h_span
    ] = patch
    fixate_belief[
        loc[0]: loc[0] + 2 * w_span,
        loc[1]: loc[1] + 2 * h_span
    ] = fixate_patch
f, axes = plt.subplots(1, 3, sharex=True, sharey=True)
axes[1].scatter(*(saccades.T + w_span), color='white')
axes[0].imshow(padded_image)
axes[0].set_title('True')
axes[0].axis('off')
axes[1].imshow(belief)
axes[1].set_title('Perceived')
axes[1].axis('off')
axes[2].imshow(fixate_belief)
axes[2].set_title('Fixation (w/o saccade)')
axes[2].axis('off')
axes[2].scatter(*(fixate_saccades.T + w_span), color='white')

plt.savefig('test.png', dpi=300)
