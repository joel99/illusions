# Model definition
from typing import List, Tuple
from yacs.config import CfgNode as CN

import torch
from torch import nn, optim
import torch.nn.functional as F
import pytorch_lightning as pl

class SaccadingRNN(pl.LightningModule):
    # We'll train this network with images. We can saccade for e.g. 100 timesteps per image and learn through self-supervision.
    # This model is driven by a saccading policy (here it is hardcoded to pick a random sequence of locations)
    # The model is trained to predict the next visual input.
    # 3 loss schemes
    # 1. E2E, deconv RNN state to predict pixels
    # TODO 2. Local Predictive Coding: RNN predicts CNN output, CNN predicts pixels (i.e. through autoencoding)
    # TODO 3. Supervised: Like E2E, but with random locations as input.

    def __init__(
        self,
        config: CN,
    ):
        self.cfg = config.MODEL
        assert self.cfg.TYPE is 'gru', 'non-gru rnns unsupported'
        assert self.cfg.ADAPTATION_LAYER is False, 'adaptation not supported (yet)' # TODO implement
        super().__init__()
        # TODO add Gaussian mask or something for FOV
        self.adaptation = None
        self.cnn_sensory = nn.CNN(None) # TODO + also include flattening here
        self.rnn = nn.GRU(self.cfg.SENSORY_SIZE, self.cfg.HIDDEN_SIZE, 1)
        self.cnn_predictive = nn.CNN(None) # TODO

        self.criterion = nn.MSELoss() # TODO is there something better than MSE?
        self.weight_decay = config.TRAIN.WEIGHT_DECAY

    def _generate_random_saccades(self, image) -> List[Tuple]:
        # TODO update with additional parameters i.e. modes to support "fixation"
        # returns:
        # - a list of valid focal coordinates
        # TODO implement
        # TODO figure out a way to normalize these coordinates
        return []

    def sense(self, image, saccade_focus):
        # Apply FOV
        view = image # TODO do something with the saccade
        if self.adaptation is not None:
            view = self.adaptation(view)
        return view

    def forward(self, image, saccade_focus, hidden_state):
        # Step the perception
        # args:
        #   image - [B x H x W]
        #   saccade_focus - [B x 2]
        #   hidden_state - [B x Hidden]
        # returns:
        #   hidden_states - [B x Hidden]
        view = self.sense(image, saccade_focus)
        cnn_view = self.cnn_sensory(view)
        # TODO concat propioception/saccade
        # Encode
        outputs, hidden_state = self.rnn(cnn_view, hidden_state)
        patches = self.cnn_predictive(outputs) # deconv so we can apply loss
        return patches, view, cnn_view, hidden_state

    def saccade_image(self, image):
        # Self-supervised saccading sequence
        hidden_state = torch.zeros((1, image.size(0), self.cfg.HIDDEN_SIZE), device=self.device) # Init
        saccades = self._generate_random_saccades(image)
        all_views = []
        all_patches = []
        for saccade in saccades:
            patches, view, cnn_view, hidden_state = self(image, saccade, hidden_state)
            all_views.append(view)
            all_patches.append(patches)
            # TODO get loss (the above is probably off by 1)
        loss = self.criterion(torch.stack(all_views), torch.stack(all_patches))
        return loss, hidden_state, all_views, all_patches

    # Pytorch lightning API below

    def predict(self, image):
        # arg: image - H x W
        pass

    def training_step(self, batch, batch_idx):
        # batch - B x H x W
        loss = self.saccade_image(batch)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.saccade_image(batch)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, hidden_state, all_views, all_patches = self.saccade_image(batch)
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