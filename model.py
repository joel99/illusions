# Model definition
from typing import List, Tuple, Optional
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
        assert self.cfg.ADAPTATION_LAYER is False, 'adaptation not supported (yet)' # TODO Taylor implement
        super().__init__()
        self.adaptation = None # TODO Taylor implement
        conv_dim = self.cfg.CONV_CHANNELS
        # self.conv_outw = self.conv_outh = 4 # Manually checked
        self.conv_outw = self.conv_outh = 2 # Manually checked

        def gen_activation():
            return nn.ReLU() if self.cfg.ACTIVATION is 'relu' else nn.LeakyReLU(0.05)

        self.cnn_sensory = nn.Sequential(
            conv(config.TASK.CHANNELS, conv_dim, 4),
            gen_activation(),
            conv(conv_dim, conv_dim, 4),
            gen_activation(),
            conv(conv_dim, conv_dim * 2, 4),
            gen_activation(),
            conv(conv_dim * 2, conv_dim, 4),
            gen_activation(),
        )
        flat_cnn_out = conv_dim * self.conv_outh * self.conv_outw

        if self.cfg.INCLUDE_PROPRIO:
            memory_input_size = self.cfg.SENSORY_SIZE - PROPRIOCEPTION_SIZE
        else:
            memory_input_size = self.cfg.SENSORY_SIZE
        self.sensory_to_memory = nn.Linear(flat_cnn_out, memory_input_size)

        if self.cfg.INCLUDE_PROPRIO:
            sensory_prediction_size = self.cfg.HIDDEN_SIZE + PROPRIOCEPTION_SIZE
        else:
            sensory_prediction_size = self.cfg.HIDDEN_SIZE

        # TODO, memory can either predict CNN directly (we should implement this first)
        # Or we can use an intermediate flat vector (TBD)
        # if 'predictive' in self.cfg.OBJECTIVES:
        #     self.sensory_to_memory = nn.Linear(flat_cnn_out, flat_cnn_in) # mock memory for reconstruction.

        # Extract cnn_view prediction from memory
        self.predict_sensory = nn.Sequential(
            nn.Linear(sensory_prediction_size, self.cfg.HIDDEN_SIZE),
            gen_activation(),
            nn.Linear(self.cfg.HIDDEN_SIZE, flat_cnn_out),
            nn.Unflatten(-1, (conv_dim, self.conv_outh, self.conv_outw))
        )
        self.rnn = nn.GRU(self.cfg.SENSORY_SIZE, self.cfg.HIDDEN_SIZE, 1)

        predictive_output = config.TASK.CHANNELS if self.cfg.QUANTIZED_RECONSTRUCTION <= 1 else self.cfg.QUANTIZED_RECONSTRUCTION
        if self.cfg.UPSAMPLE_CONV:
            self.cnn_predictive = nn.Sequential(
                upsample_conv(conv_dim, conv_dim * 2, 2),
                gen_activation(),
                upsample_conv(conv_dim * 2, conv_dim, 2),
                gen_activation(),
                upsample_conv(conv_dim, conv_dim, 2),
                gen_activation(),
                upsample_conv(conv_dim, predictive_output, 2),
                nn.Tanh()
            )
        else:
            self.cnn_predictive = nn.Sequential(
                deconv(conv_dim, conv_dim * 2, 4),
                gen_activation(),
                deconv(conv_dim * 2, conv_dim, 4),
                gen_activation(),
                deconv(conv_dim, conv_dim, 4),
                gen_activation(),
                deconv(conv_dim, predictive_output, 4),
                nn.Tanh()
            )
        self.mse = nn.MSELoss()
        self.xent = nn.CrossEntropyLoss()
        self.weight_decay = config.TRAIN.WEIGHT_DECAY
        self.saccade_training_mode = self.cfg.SACCADE
        self.view_mask = self._generate_falloff_mask()

    def _evaluate_image_pred(self, gt, pred):
        # Note: the actual quantization scheme was proposed in a more sophisticated way (ref: https://github.com/richzhang/colorization/tree/caffe)
        # Doing this for simplicity
        # gt: T B C H W
        if self.cfg.QUANTIZED_RECONSTRUCTION <= 1:
            return self.mse(pred, gt)
        # Expecting GT to vary [-0.5, 0.5]
        quantized_gt = ((gt + 0.5) * (self.cfg.QUANTIZED_RECONSTRUCTION - 1)).long().flatten(0, 2)
        return self.xent(pred.flatten(0, 1), quantized_gt)

    def _quantized_pred(self, pred):
        if self.cfg.QUANTIZED_RECONSTRUCTION <= 1:
            return pred
        # Expecting GT to vary [-0.5, 0.5]
        with torch.no_grad():
            quantized_pred = torch.argmax(pred, dim=2).float() # T B H W varying 0 - 7
            quantized_pred = quantized_pred / (self.cfg.QUANTIZED_RECONSTRUCTION - 1) # asymmetry is nervewracking
            BIN_SPAN = 0.5 / self.cfg.QUANTIZED_RECONSTRUCTION
            quantized_pred = quantized_pred - 0.5 + BIN_SPAN # T B H W
            return quantized_pred.unsqueeze(-3) # T B C H W

    def _generate_saccades(self, image, length=50, mode=None) -> torch.tensor:
        # Generate a sequence of saccading focal coordinates.
        # * Does not provide margin, i.e. coordinates can be on corner of image.
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
        elif mode == 'constant':
            coords_ratio = torch.full((length, 2), 0.5, device=self.device)
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

    def _predict_at_location(self, state, focus, mode='patch'):
        r"""
        Memory conditioned predictions
        Args:
            state: [T x B x H] state to make prediction with (either CNN output in predictive coding, or memory)
            focus: [T x 2] location to predict at (given as absolute ratio)
            mode: 'patch' for pixel reconstruction or 'percept' for CNN reconstruction
        Returns:
            patches: [B x C x H x W] predicted patches at given locations
        """
        if self.cfg.INCLUDE_PROPRIO:
            state = torch.cat([
                state, focus.unsqueeze(1).expand(-1, state.size(1), 2)
            ], dim=-1)

        prediction_cue = self.predict_sensory(state) # extract cnn prediction
        if mode == 'patch': # Make a prediction about image patch from memory
            # T x B x Hidden -> T x B x C x H x W
            return self.cnn_predictive(prediction_cue.flatten(0, 1)).unflatten(
                0, (state.size(0), state.size(1))
            )
        else:
            return prediction_cue

    def forward(self, view, saccade_focus, hidden_state):
        r"""
            Roll forward perception.
            For efficiency, this is a batch update.
            Requires some reshaping for a single step (and thus may be annoying for active perception)
            Args:
                view: [T x B x C x H x W] - cropped views.
                saccade_focus: [T x 2] - proprioception (shared across batch)
                hidden_state: [B x Hidden] (initial hidden state)
            Returns:
                cnn_view: [T x B x C x H' x W'] CNN outputs
                memory_prediction: [T x B x [C x H' x W']] RNN outputs
                hidden_state: [B x Hidden] Final hidden state
        """
        if self.adaptation is not None:
            view = self.adaptation(view)

        cnn_view = self.cnn_sensory(view.flatten(0, 1)).unflatten(
            0, (view.size(0), view.size(1)) # T x B x C x H x W
        )
        flat_cnn_view = cnn_view.flatten(2, -1) # T x B x H
        memory_input = self.sensory_to_memory(flat_cnn_view) # T x B x Hidden
        if self.cfg.INCLUDE_PROPRIO:
            saccade_sense = saccade_focus.unsqueeze(1).expand(
                -1, view.size(1), -1
            ) # T x B x 2
            memory_input = torch.cat([memory_input, saccade_sense], dim=-1)
        rnn_view, hidden_state = self.rnn(memory_input, hidden_state) # T x B x H

        return cnn_view, flat_cnn_view, rnn_view, hidden_state

    def get_views(self, image, saccades):
        all_views = [] # raw and noised.
        w_span, h_span = self.cfg.FOV_WIDTH // 2, self.cfg.FOV_HEIGHT // 2

        # 1a. Window selection.
        # TODO Joel this can probably be vectorized
        # If anyone else knows how to do this please go ahead + cc: Joel.
        padded_image = F.pad(image, (w_span, w_span, h_span, h_span))
        for saccade in saccades:
            w_span, h_span = self.cfg.FOV_WIDTH // 2, self.cfg.FOV_HEIGHT // 2
            view = padded_image[
                ...,
                saccade[0]: saccade[0] + 2 * w_span, # centered around saccade + span
                saccade[1]: saccade[1] + 2 * h_span, # to account for padding
            ]
            all_views.append(view)
        all_views = torch.stack(all_views, 0) # T x B x C x H x W
        noise = torch.randn(all_views.size(), device=self.device) * self.cfg.FOV_FALLOFF
        noised_views = all_views + noise * self.view_mask
        if self.cfg.CLAMP_FOV:
            noised_views = torch.clamp(noised_views, -1, 1) # this bound is pretty aribtrary

        return all_views, noised_views

    def get_proprioception(self, image, saccades):
        if self.cfg.PROPRIOCEPTION_DELTA:
            proprioception = (saccades[1:] - saccades[:-1]).float()
            return torch.cat([
                torch.zeros((1,2), dtype=torch.float, device=self.device), proprioception
            ], dim=0)
        else:
            return saccades.float() / torch.tensor(image.size()[-2:], device=self.device).float() - 0.5 # 0 center

    def saccade_image(
        self,
        image,
        length=50,
        mode=None,
        initial_state: Optional[torch.tensor]=None,
    ):
        r"""
            General purpose saccading saccading sequence.
            Returns will vary based on saccading mode.

            Args:
                image: [B x C x H x W] Image batch to saccade over
                length: length of saccade
                mode: saccading mode (see `_generate_saccades`)
                initial_state: [B x H] Initial state for memory
            Returns:
                loss
                views: [T x B x C x H x W] (model observations)
                patches: [T x B x C x H x W] (model percepts).
                hidden_state: [B x H] (model memory at conclusion)
        """
        self.view_mask = self.view_mask.to(self.device) # ! weird, we can't set it in __init__()

        if initial_state is None:
            hidden_state = torch.zeros((1, image.size(0), self.cfg.HIDDEN_SIZE), device=self.device)
        else:
            hidden_state = initial_state

        # Generate observations
        saccades = self._generate_saccades(image, length=length, mode=mode)
        all_views, noised_views = self.get_views(image, saccades)
        proprioception = self.get_proprioception(image, saccades)

        # Forward and calculate loss
        cnn_view, flat_cnn_view, rnn_view, hidden_state = self(noised_views, proprioception, hidden_state)
        losses = []
        all_patches = []
        supervisory_view = noised_views if self.cfg.NOISED_SIGNAL else all_views
        for objective in self.cfg.OBJECTIVES:
            if objective == 'predictive_patch':
                all_patches = self._predict_at_location(rnn_view[:-1], saccades[1:], mode='patch')
                loss = self._evaluate_image_pred(supervisory_view[1:], all_patches)
            elif objective == 'autoencode': # Legacy, debug objective
                if self.cfg.REACTIVE:
                    all_patches = self.cnn_predictive(cnn_view.flatten(0, 1)).unflatten(0, (cnn_view.size(0), cnn_view.size(1)))
                else:
                    all_patches = self._predict_at_location(rnn_view, saccades) # Model must reproduce exactly what it saw.
                loss = self._evaluate_image_pred(supervisory_view, all_patches)
            elif objective == 'predictive':
                # Each module predicts its input (implying autoencoding for CNN)
                all_patches = self.cnn_predictive(
                    cnn_view.flatten(0, 1)
                ).unflatten(0, (cnn_view.size(0), cnn_view.size(1)))
                loss_cnn = self._evaluate_image_pred(supervisory_view, all_patches)
                cnn_predictions = self._predict_at_location(rnn_view[:-1], saccades[1:], mode='percept')
                loss_rnn = self.mse(cnn_view[1:], cnn_predictions)
                loss = loss_cnn + loss_rnn
            elif objective == 'random':
                raise NotImplementedError # TODO Joel
            else:
                raise Exception
            losses.append(loss)
        loss = torch.stack(losses).sum() # no tradeoff terms

        # TODO how do we decide what patches to return?
        # Currently it'll return the last objective's patches
        return loss, all_views, noised_views, all_patches, hidden_state

    # Pytorch lightning API below
    # TODO systematic prediction to get a "full image" percept.
    # Problem is that we don't train in any systematic matter
    # So grid-based reconstruction is probably not exactly accurate.

    @torch.no_grad()
    def predict(
        self,
        image,
        saccade_first=True,
        mode='autoencode'
    ):
        r"""
            Will take the image, saccade randomly over it, and then make predictions (on a different random saccade sequence).
            Args:
                image: C x H x W.
        """
        loss, all_views, noised_views, all_patches, state = self.saccade_image(image.unsqueeze(0))
        print('First saccade:', loss)
        if saccade_first: # ok... what now.
            loss, all_views, noised_views, all_patches, state = \
            self.saccade_image(image.unsqueeze(0), initial_state=state)
            print('Next saccade:', loss)

        if self.cfg.QUANTIZED_RECONSTRUCTION:
            all_patches = self._quantized_pred(all_patches)
        return all_views, noised_views, all_patches, state

    @torch.no_grad()
    def predict_with_saccades(
        self,
        image,
        saccades,
        initial_state=None,
        mode='predictive_patch'
    ):
        r"""
            Finer control
        """
        image = image.unsqueeze(0)
        all_patches = []

        if initial_state is None:
            hidden_state = torch.zeros((1, image.size(0), self.cfg.HIDDEN_SIZE), device=self.device)
        else:
            hidden_state = initial_state
        # Generate observations
        all_views, noised_views = self.get_views(image, saccades)
        proprioception = self.get_proprioception(image, saccades)

        cnn_view, flat_cnn_view, rnn_view, hidden_state = self(noised_views, proprioception, hidden_state)
        if mode == 'predictive_patch':
            all_patches = self._predict_at_location(rnn_view[:-1], saccades[1:], mode='patch')
        elif mode == 'autoencode': # Legacy, debug objective
            all_patches = self.cnn_predictive(
                cnn_view.flatten(0, 1)
            ).unflatten(0, (cnn_view.size(0), cnn_view.size(1)))
        return all_views, noised_views, all_patches, hidden_state

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
        loss, *_ = self.saccade_image(batch)
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

def upsample_conv(c_in, c_out, scale_factor, k_size=3, stride=1, pad=1, bn=True):
    return nn.Sequential(
        nn.Upsample(scale_factor=scale_factor),
        nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=False),
        nn.BatchNorm2d(c_out)
    )

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
