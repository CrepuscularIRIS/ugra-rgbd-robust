"""Datasets for the DFormer family (NYU Depth V2: 795/654, SUN RGB-D: 5,285/5,050), reusing the upstream classes.

Layout under $RGBD_DATA_ROOT:  NYUDepthv2/{RGB,Depth,Label,train.txt,test.txt}  and the same for SUNRGBD/.
(The DFormer authors' packaged zips are on Hugging Face: bbynku/DFormerv2, datasets/NYUDepthv2.zip, SUNRGBD.zip.)
Batches are dicts with "data" (RGB), "modal_x" (standardized depth, c = 0 means empty), "label".
"""
from __future__ import annotations

from torch.utils.data import DataLoader, Dataset


def _setting(cfg) -> dict:
    return {
        "rgb_root": cfg.rgb_root_folder, "rgb_format": cfg.rgb_format,
        "gt_root": cfg.gt_root_folder, "gt_format": cfg.gt_format, "transform_gt": cfg.gt_transform,
        "x_root": cfg.x_root_folder, "x_format": cfg.x_format, "x_single_channel": cfg.x_is_single_channel,
        "class_names": cfg.class_names, "train_source": cfg.train_source, "eval_source": cfg.eval_source,
        "dataset_name": cfg.dataset_name, "backbone": cfg.backbone,
    }


def train_dataset(host) -> Dataset:
    """Training split with the upstream augmentation (random mirror, scale, crop) and standardization."""
    from utils.dataloader.dataloader import TrainPre
    from utils.dataloader.RGBXDataset import RGBXDataset

    cfg = host.cfg
    return RGBXDataset(_setting(cfg), "train", TrainPre(cfg.norm_mean, cfg.norm_std, cfg.x_is_single_channel, cfg))


def eval_dataset(host, split: str = "val") -> Dataset:
    """Test split ("val") or the un-augmented training images ("train", used as a calibration set)."""
    from utils.dataloader.dataloader import ValPre
    from utils.dataloader.RGBXDataset import RGBXDataset

    cfg = host.cfg
    return RGBXDataset(_setting(cfg), split, ValPre(cfg.norm_mean, cfg.norm_std, cfg.x_is_single_channel, cfg))


def train_loader(host, batch_size: int | None = None, num_workers: int = 6) -> DataLoader:
    return DataLoader(train_dataset(host), batch_size=batch_size or host.cfg.batch_size, shuffle=True,
                      num_workers=num_workers, drop_last=True, pin_memory=True, persistent_workers=num_workers > 0)


def eval_loader(host, split: str = "val", num_workers: int = 4) -> DataLoader:
    return DataLoader(eval_dataset(host, split), batch_size=1, shuffle=False, num_workers=num_workers)
