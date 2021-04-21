#!/usr/bin/env python3
# Author: Joel Ye

from typing import List, Optional, Union

from yacs.config import CfgNode as CN

DEFAULT_CONFIG_DIR = "config/"
CONFIG_FILE_SEPARATOR = ","

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()
_C.SEED = 0

# Name of experiment
_C.VARIANT = "test"

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
_C.TASK = CN()
_C.TASK.NAME = 'UNIFORMITY'
_C.TASK.OVERFIT = False # Debug

_C.TASK.UNIFORMITY = CN()
_C.TASK.CHANNELS = 3

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
# * Note, many of these settings are untested.
_C.MODEL = CN()
_C.MODEL.TYPE = 'gru'
_C.MODEL.ACTIVATION = 'lrelu' # lrelu, relu, sin. Note, sin is more prone to overfit.
_C.MODEL.CONV_CHANNELS = 32
_C.MODEL.SENSORY_SIZE = 64 # input size to RNN
_C.MODEL.HIDDEN_SIZE = 128
_C.MODEL.ADAPTATION_LAYER = False # Include adaptation or not.
_C.MODEL.FOV_FALLOFF = 0.001 # Scale factor for noise strength as a function of distance to focus. Noise = FOV_FALLOFF * N(0, r^2)
_C.MODEL.FOV_WIDTH = 32 # FOV in spanned pixels (not angles). Should be set such that noise roughly overwhelms signal at edge
_C.MODEL.FOV_HEIGHT = 32
_C.MODEL.TRANSITION_MLP = False
_C.MODEL.CLAMP_FOV = True # Clamp FOV output to [-1, 1]
_C.MODEL.SACCADE = 'walk'
_C.MODEL.WALK_PACE = 0.1
_C.MODEL.PROPRIOCEPTION_DELTA = False
_C.MODEL.FOURIER_PROPRIO = False
_C.MODEL.FOURIER_PROPRIO_SCALE = 1 # unused
_C.MODEL.UPSAMPLE_CONV = True
_C.MODEL.INCLUDE_PROPRIO = True # debug option to flag all proprioceptive inputs (vision-only model)
_C.MODEL.REACTIVE = False # debug option to turn off RNN. Assumes include_proprio.
_C.MODEL.QUANTIZED_RECONSTRUCTION = 0 # 0 - use MSE. N > 1 => quantize into N bins.
_C.MODEL.OBJECTIVES = ['predictive']
# predictive: Predict error terms (autoencode + percept prediction)
# predictive_patch: predict next view prior to seeing it (on RNN)
# random: predict random views (Not implemented) - probably should support warmup period to see image.
# autoencode: predict current state well (on RNN)
# adversarial_patch: real-fake at patch level
# contrast: minibatch contrast, same image = positives. to improve image reprs.
# * the problem is that we can't remember.
_C.MODEL.NOISED_SIGNAL = True # Whether to use noised patches for supervision. True is self-supervised.

# -----------------------------------------------------------------------------
# Train Config
# -----------------------------------------------------------------------------
_C.TRAIN = CN()

_C.TRAIN.BATCH_SIZE = 64 # 32-64 are approx same results
_C.TRAIN.EPOCHS = 10
_C.TRAIN.WEIGHT_DECAY = 0.0


def get_cfg_defaults():
  """Get default LFADS config (yacs config node)."""
  return _C.clone()

def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CN:
    r"""Create a unified config with default values overwritten by values from
    :p:`config_paths` and overwritten by options from :p:`opts`.

    :param config_paths: List of config paths or string that contains comma
        separated list of config paths.
    :param opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example,
        :py:`opts = ['FOO.BAR', 0.5]`. Argument can be used for parameter
        sweeping or quick tests.
    """
    config = get_cfg_defaults()
    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    if opts:
        config.merge_from_list(opts)

    config.freeze()
    return config

