import os.path as osp
import argparse
from yacs.config import CfgNode as CN

import torch
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor

from config.default import get_config
from dataset import UniformityDataset
from model import SaccadingRNN

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', '-c', type=str, required=True)

    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=0,
    )

    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    return parser

def run_exp(
    config: str,
    seed: int,
    opts=None,
):
    # Grab config, set name, all that good stuff.
    if opts is None:
        opts = []
    variant_name = osp.split(config)[1].split('.')[0]
    config = get_config(config, opts)
    config.defrost()
    config.VARIANT = variant_name
    config.SEED = seed
    config.freeze()

    pl.utilities.seed.seed_everything(seed=seed)
    print(f"Starting {config.VARIANT} run with seed {config.SEED}")

    model = SaccadingRNN(config)
    print(model)

    if config.TASK.NAME == 'UNIFORMITY':
        dataset = UniformityDataset(config, split="train")
    else:
        raise NotImplementedError

    length = len(dataset)
    train, val = random_split(
        dataset,
        [int(length * 0.8), length - int(length * 0.8)],
        generator=torch.Generator().manual_seed(42)
    )
    print("Training on ", len(train), " examples")

    lr_logger = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        max_epochs=config.TRAIN.EPOCHS,
        gpus=1,
        val_check_interval=1.0,
        callbacks=[lr_logger],
        default_root_dir=f"./runs/{config.VARIANT}-{config.SEED}"
    )

    trainer.fit(
        model,
        DataLoader(train, batch_size=config.TRAIN.BATCH_SIZE), # * Note, there's 2x the number of minibatches that I expect, not sure why.
        DataLoader(val, batch_size=config.TRAIN.BATCH_SIZE)
    )

    print()
    print("Train results")
    trainer.test(model, DataLoader(dataset, batch_size=64))

    if config.TASK.NAME == 'UNIFORMITY':
        test_dataset = UniformityDataset(config, split="test")
    else:
        raise NotImplementedError

    print()
    print("Test results")
    trainer.test(model, DataLoader(test_dataset, batch_size=64))


def main():
    parser = get_parser()
    args = parser.parse_args()
    run_exp(**vars(args))

if __name__ == "__main__":
    main()
