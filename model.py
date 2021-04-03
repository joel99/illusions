# Model definition
from typing import List, Tuple
from yacs.config import CfgNode as CN

import torch
from torch import nn, optim
import torch.nn.functional as F
import pytorch_lightning as pl

PROPRIOCEPTION_SIZE = 2
class SaccadingRNN(pl.LightningModule):
    # We'll train this network with images. We can saccade for e.g. 100 timesteps per image and learn through self-supervision.
    # This model is driven by a saccading policy (here it is hardcoded to pick a random sequence of locations)
    # The model is trained to predict the next visual input.
    # 3 loss schemes
    # 1. E2E, deconv RNN state to predict pixels
    # TODO Joel Local Predictive Coding: RNN predicts CNN output, CNN predicts pixels (i.e. through autoencoding)
    # TODO 3. Supervised: Like E2E, but with random locations as input.

    def __init__(
        self,
        config: CN,
    ):
        self.cfg = config.MODEL
        assert self.cfg.TYPE is 'gru', 'non-gru rnns unsupported'
        assert self.cfg.ADAPTATION_LAYER is False, 'adaptation not supported (yet)' # TODO implement
        super().__init__()
        self.adaptation = None # TODO Taylor implement
        conv_dim = self.cfg.CONV_CHANNELS
        self.conv_outw = 4 # Manually checked
        self.conv_outh = 4
        # 3 layer CNN
        self.cnn_sensory = nn.Sequential(
            conv(config.TASK.CHANNELS, conv_dim, 4),
            nn.LeakyReLU(0.05),
            conv(conv_dim, conv_dim * 2, 4),
            nn.LeakyReLU(0.05),
            conv(conv_dim * 2, conv_dim, 4),
            nn.LeakyReLU(0.05),
        )
        self.flatten_sensory = nn.Sequential(
            nn.Flatten(start_dim=-3),
            nn.Linear(conv_dim * self.conv_outh * self.conv_outw, self.cfg.SENSORY_SIZE - PROPRIOCEPTION_SIZE),
        )

        self.predict_sensory = nn.Sequential(
            nn.Linear(self.cfg.HIDDEN_SIZE + PROPRIOCEPTION_SIZE, self.cfg.HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(self.cfg.HIDDEN_SIZE, self.conv_outh * self.conv_outw) # no nn.unflatten
        )
        self.rnn = nn.GRU(self.cfg.SENSORY_SIZE, self.cfg.HIDDEN_SIZE, 1)
        self.cnn_predictive = nn.Sequential(
            deconv(1, conv_dim * 2, 4),
            nn.LeakyReLU(0.05),
            deconv(conv_dim * 2, conv_dim, 4),
            nn.LeakyReLU(0.05),
            deconv(conv_dim, config.TASK.CHANNELS, 4),
            nn.Tanh()
        )
        self.criterion = nn.MSELoss() # TODO can we improve on MSE?
        self.weight_decay = config.TRAIN.WEIGHT_DECAY
        self.saccade_training_mode = self.cfg.SACCADE
        self.view_mask = self._generate_falloff_mask()

    def _generate_saccades(self, image, length=50, mode=None) -> torch.tensor:
        # Generate a sequence of saccading focal coordinates.
        # * Does not provide margin, i.e. coordinates can be on corner of image. TBD whether this should change
        # * We allow this so model can observe corner accurately.
        # args:
        #   image: [* x H x W]. image or image batch to saccade over.
        #   length: length of saccading sequence
        #   mode: saccadding mode. (controlling model operation)
        # returns:
        #   coords: [length x 2] . pixel coordinates
        if mode is None:
            mode = self.saccade_training_mode
        H, W = image.size()[-2:]

        if mode == 'random': # uniform distribution
            # ! Untested
            coords_ratio = torch.rand((length, 2), device=image.device)
        elif mode == 'walk': # random walk
            # TODO Make walking more realistic -- humans don't random walk off the image.
            # A better heuristic is to weigh directional probability by location
            #   but that might be too slow.
            WALK_PACE = 0.1
            start_ratio = torch.tensor([0.5, 0.5], device=self.device).unsqueeze(0).float() # 1 x 2
            deltas_ratio = torch.randn((length - 1, 2), device=self.device) * WALK_PACE
            coords_ratio = torch.cat([start_ratio, deltas_ratio], dim=0)
            coords_ratio = torch.cumsum(coords_ratio, dim=0)
        elif mode == 'fixate': # center with Gaussian offset for drift.
            # ! Untested
            start = torch.tensor([0.5, 0.5], device=self.device).unsqueeze(0).float() # 1 x 2
            DRIFT = 0.05 # 0.05 of a 128 x 128 would constrain to a 6x6 range, seems reasonable.
            # TODO Chris - add config to control amount of drift
            # TODO Chris - calculate an appropriately calibrated fixation radius based on human measurements (at level of saccadic drift)
            deltas = torch.randn((length, 2), device=self.device) * DRIFT
            coords_ratio = (start + deltas) * torch.tensor
        coords_ratio = torch.clamp(coords_ratio, 0, 1)
        image_scale = torch.tensor([H, W], device=self.device).float()
        return (coords_ratio * image_scale).long()

    def _generate_falloff_mask(self):
        # TODO Joel look at what Gaussian blur actually is.
        # Construct FOV falloff (H x W)
        # Since we only construct this once, inefficient manual construction is fine
        mask = torch.zeros(
            self.cfg.FOV_WIDTH, self.cfg.FOV_HEIGHT, device=self.device
        ).float()
        center_h = (mask.size(-2) - 1) / 2.
        center_w = (mask.size(-1) - 1) / 2.
        for h in range(mask.size(-2)):
            for w in range(mask.size(-1)):
                mask[h,w] = (h - center_h) ** 2 + (w - center_w) ** 2
        return mask

    def forward(self, view, saccade_focus, hidden_state):
        # Step the perception
        # * TODO Joel - we can feed a whole sequence of saccades for increased efficiency.
        # ? I wonder why it's so slow absent ^?
        # args:
        #   image: [B x C x H x W] - CROPPED
        #   saccade_focus: [2] - proprioception (TODO support [T x 2])
        #   hidden_state: [B x Hidden]
        # returns:
        #   hidden_state: [B x Hidden]

        # TODO understand the loss itself.
        if self.adaptation is not None:
            view = self.adaptation(view)
        cnn_view = self.cnn_sensory(view)
        flat_view = self.flatten_sensory(cnn_view)

        # Right now we are autoencoding -- we feed a view, and immediately predict it.
        # It is worth seeing how well this does, I guess.
        # TODO test cue prior to the CNN receiving it. Right now it's acting as autoencoder.
        saccade_sense = saccade_focus.expand(flat_view.size(0), -1)
        sense = torch.cat([flat_view, saccade_sense], dim=-1)
        outputs, hidden_state = self.rnn(sense.unsqueeze(0), hidden_state) # B x H
        outputs = outputs.squeeze(0)
        prediction_cue = torch.cat([outputs, saccade_sense], dim=-1)
        prediction_cue = self.predict_sensory(prediction_cue)
        prediction_cue = prediction_cue.reshape(view.size(0), -1, self.conv_outh, self.conv_outw)
        patches = self.cnn_predictive(prediction_cue) # deconv so we can apply loss
        return patches, cnn_view, hidden_state

    def saccade_image(self, image):
        # * Weakly-supervised saccading sequence
        # TODO Joel support other modes
        # args:
        #   image: [B x C x H x W]
        self.view_mask = self.view_mask.to(self.device) # ! weird, we can't set it in __init__()

        hidden_state = torch.zeros((1, image.size(0), self.cfg.HIDDEN_SIZE), device=self.device) # Init
        saccades = self._generate_saccades(image)
        all_views = [] # raw and unnoised. # TODO Joel consider whether we should be predicting noised images.
        all_patches = []
        w_span, h_span = self.cfg.FOV_WIDTH // 2, self.cfg.FOV_HEIGHT // 2
        image = F.pad(image, (w_span, w_span, h_span, h_span)) # zero-pads edges
        last_saccade = saccades[0]
        for saccade in saccades:
            # Window
            w_span, h_span = self.cfg.FOV_WIDTH // 2, self.cfg.FOV_HEIGHT // 2
            view = image[
                ...,
                saccade[0]: saccade[0] + 2 * w_span, # centered around saccade + span
                saccade[1]: saccade[1] + 2 * h_span, # to account for padding
            ]
            all_views.append(view)
            # Noise
            noise = torch.randn(view.size(), device=self.device) * self.cfg.FOV_FALLOFF
            view = view + noise * self.view_mask
            if self.cfg.CLAMP_FOV:
                view = torch.clamp(view, -1, 1)
            if self.cfg.PROPRIOCEPTION_DELTA:
                proprioception = saccade - last_saccade
                last_saccade = saccade
            else:
                proprioception = saccade / torch.tensor(image.size()[-2:], device=self.device)
            # We're here right now.
            patches, cnn_view, hidden_state = self(view, proprioception, hidden_state)
            all_patches.append(patches)
            # TODO get loss (the above is probably off by 1)
        loss = self.criterion(torch.stack(all_views), torch.stack(all_patches))
        return loss, all_views, all_patches, hidden_state

    # Pytorch lightning API below

    def predict(self, image):
        # arg: image - H x W
        pass

    def training_step(self, batch, batch_idx):
        # batch - B x H x W
        loss, *_ = self.saccade_image(batch)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, *_ = self.saccade_image(batch)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, all_views, all_patches, hidden_state = self.saccade_image(batch)
        self.log('test_loss', loss, prog_bar=True)
        # TODO probably want to log down images as well
        return loss

    def configure_optimizers(self):
        # Reduce LR on plateau as a reasonable default
        optimizer = optim.Adam(self.parameters(), lr=1e-3, weight_decay=self.weight_decay)
        return {
            'optimizer': optimizer,
            'lr_scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=50),
            'monitor': 'val_loss'
        }

def deconv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom deconvolutional layer for simplicity."""
    layers = []
    layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad, bias=False))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)

def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom convolutional layer for simplicity."""
    layers = []
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=False))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)
