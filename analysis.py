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

seed = 0

config = './config/base.yaml'
config = './config/debug.yaml'
# config = './config/debug2.yaml'
# config = './config/base.yaml'
# config = './config/base_e2e.yaml'
config = './config/large.yaml'
config = './config/large_adv.yaml'
version = 5
version = 0
version = 2
config = './config/large_adv_sin.yaml'
version = 4

config = './config/large_sin.yaml'
# version = 18
version = 19
# config = './config/snakes_slow.yaml'
# version = 0
# version = 2
# version = 3
config = './config/snakes.yaml'
version = 9

config = './config/snakes_large.yaml'
version = 0

config = './config/snakes_cifar.yaml'
version = 6
# TODO add state_dict
config = './config/snakes_ssim.yaml'
version = 3

config = './config/pdi.yaml'
version = 0
# version = 3
config = './config/pdi_polar.yaml'
version = 7

config = './config/circ.yaml'
version = 0

config = './config/troxler_batch_clean.yaml'
config = './config/troxler_batch.yaml'
version = 0

# config = './config/pdi_fourier.yaml'
# config ='./config/pdi_fourier_noise.yaml'
# config = './config/pdi_fourier_random.yaml'
config = './config/pdi_fourier_noise_random.yaml'
version = 0 # Overfit
# version = 2 # Generalization

variant = osp.split(config)[1].split('.')[0]
config = get_config(config)

root = Path(f'runs/{variant}-{seed}/lightning_logs/')
# * This is the default output, if you want to play around with a different checkpoint load it here.
model_ckpt = list(root.joinpath(f"version_{version}").joinpath('checkpoints').glob("*"))[0]

weights = torch.load(model_ckpt, map_location='cpu')
model = SaccadingRNN(config)
model.load_state_dict(weights['state_dict'])
model.eval()

if config.TASK.NAME == 'UNIFORMITY':
    dataset = UniformityDataset(config, split="train")
else:
    dataset = UniformistyDataset(config, split="train", dataset_root=f'./data/{config.TASK.NAME}', augment=['rotate'])
    # dataset = UniformityDataset(config, split="test", dataset_root=f'./data/{config.TASK.NAME}')

index = 0
# index = 2
# index = 9
# index = 25000

# index = 700
# index = 750
# index = 800
image = dataset[index]
proc_view = UniformityDataset.unpreprocess(image).permute(1, 2, 0)
proc_view = proc_view.squeeze(-1)
plt.imshow(proc_view)

sac_length = 50
saccades = model._generate_saccades(image, length=sac_length)[:, :1]
all_views, noised_views, patches, state = model.predict_with_saccades(image, saccades, mode='predictive_patch')
# Note, simply using saccades twice in a row is OOD.

# all_views, noised_views, patches, state = model.predict(image)

# Hm, doesn't seem to matter.... am I looking at the right output?
# Why is my loss higher than reported?

print(all_views.size(), patches.size())
loss1 = F.mse_loss(all_views[1], patches[0])
loss1 = F.mse_loss(all_views[1:], patches)
print(loss1)
print(saccades.float().mean())
plt.imshow(image.squeeze(0))
plt.axis('off')
# Wow, there's barely any loss... what gives?

# losses = [F.mse_loss(all_views[i+1], patches[i]) for i in range(49)]
# plt.plot(losses)

#%%
# It don't even look like the right image.
times = [0, 10, 20, 30]
f, axes = plt.subplots(len(times), 2, sharex=True, sharey=True)
for i, t in enumerate(times):
    true_image = all_views[t + all_views.size(0) - patches.size(0), 0]
    # true_image = noised_views[t + all_views.size(0) - patches.size(0), 0]
    proc_true = UniformityDataset.unpreprocess(true_image).permute(1, 2, 0)
    proc_true = proc_true.squeeze(-1)

    # axes[i, 0].imshow(proc_true[..., 2])
    axes[i, 0].imshow(proc_true)
    pred_image = patches[t, 0]

    print(F.mse_loss(true_image, pred_image)) # how can my average in the previous timestep be lower than all my samples here?

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
grid_h = np.linspace(0, image.size(-2)-1, step)
grid_w = np.linspace(0, image.size(-1)-1, step)
grid_x, grid_y = np.meshgrid(grid_h, grid_w)
grid = torch.stack([torch.tensor(grid_x), torch.tensor(grid_y)], dim=-1).long()

grid = grid.flatten(0, 1)
step_state = state.expand(grid.size(0), -1, -1)

patches = model._predict_at_location(step_state, grid.unsqueeze(1), mode='patch').detach()
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
f, axes = plt.subplots(1, 2, sharex=True, sharey=True)

print(saccades.size()) # ! I think there's a transpose happening

axes[0].scatter(*(saccades.T + w_span), color='white')
# Flip direction since saccade higher = matrix lower
axes[0].imshow(padded_image, origin='lower')
axes[0].set_title('True')
axes[0].axis('off')
axes[1].imshow(belief, origin='lower')
axes[1].set_title('Perceived')
axes[1].axis('off')
plt.savefig('test.png', dpi=300)

# %%

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
