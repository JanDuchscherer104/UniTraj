import pytorch_lightning as pl
import torch

torch.set_float32_matmul_precision("medium")
import os

import hydra
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from unitraj.datasets import build_dataset
from unitraj.utils.visualization import check_loaded_data


@hydra.main(version_base=None, config_path="configs", config_name="config")
def viz(cfg):
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = OmegaConf.merge(cfg, cfg.method)

    train_set = build_dataset(cfg)
    sample = train_set[0]  # Get a sample

    fig = check_loaded_data(sample)
    # save the figure
    fig.savefig("sample.png")


if __name__ == "__main__":
    viz()
