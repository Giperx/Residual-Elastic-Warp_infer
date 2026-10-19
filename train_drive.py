#!/usr/bin/env python
"""Fine-tune TCell on driving camera pairs.

The original loop is unchanged: two images, frozen HCell, overlap L1 on the
elastic warp. Camera 5 is the reference. The other image is camera 4 or camera
3. Each epoch reshuffles a random subset of training scenes and frames.

    python train_drive.py --dataset nuscenes --gpu 1
    python train_drive.py --dataset lyft1920 --steps 20 --gpu 1
"""
import os
from argparse import ArgumentParser

_pre = ArgumentParser(add_help=False)
_pre.add_argument("--gpu", default="0")
_known, _ = _pre.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = _known.gpu

import yaml
import torch
from torch.utils.data import DataLoader

import datasets
import models
import utils
import train_tcell
from rewarp_drive.compat import install as install_gesv

install_gesv()


def leaf(loader):
    dataset = loader.dataset
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    return dataset


def make_loader(spec, tag):
    dataset = datasets.make(spec["dataset"])
    dataset = datasets.make(spec["wrapper"], args={"dataset": dataset})
    print("{} dataset: size={} pool={}".format(tag, len(dataset), len(leaf_of(dataset).pairs)), flush=True)
    loader = DataLoader(
        dataset,
        batch_size=spec["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return loader


def leaf_of(dataset):
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    return dataset


def load_pretrained(path):
    checkpoint = torch.load(path, map_location="cpu")
    h_cell = models.make(checkpoint["H_model"], load_sd=True).cuda().eval()
    t_cell = models.make(checkpoint["T_model"], load_sd=True).cuda().train()
    for parameter in h_cell.parameters():
        parameter.requires_grad = False
    return h_cell, t_cell, checkpoint["H_model"], checkpoint["T_model"]


def main():
    parser = ArgumentParser(description="Fine-tune REwarp TCell on cam5+cam4 and cam5+cam3")
    parser.add_argument("--config", default="configs/train/drive_tcell.yaml")
    parser.add_argument("--dataset", default=None, choices=["nuscenes", "lyft", "widedrive3"])
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--samples", type=int, default=None, help="random pairs per epoch")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--steps", type=int, default=0, help="run this many steps and stop; for timing")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    with open(args.config) as handle:
        config = yaml.load(handle, Loader=yaml.FullLoader)
    name = args.dataset or config.get("dataset", "nuscenes")
    defaults = {"nuscenes": 1, "lyft": 3, "widedrive3": 3}
    for spec in (config["train_dataset"], config["valid_dataset"]):
        spec["dataset"]["args"]["dataset"] = name
        if args.samples and spec is config["train_dataset"]:
            spec["dataset"]["args"]["samples_per_epoch"] = args.samples
        if args.batch_size:
            spec["batch_size"] = args.batch_size
    config["epoch_max"] = args.epochs if args.epochs else defaults.get(name, config["epoch_max"])

    save_path = os.path.join("save", "drive_" + name)
    os.makedirs(save_path, exist_ok=True)
    train_tcell.config = config
    train_tcell.log = print

    train_loader = make_loader(config["train_dataset"], "train")
    valid_loader = make_loader(config["valid_dataset"], "valid")
    h_cell, t_cell, spec_h, spec_t = load_pretrained(config["pretrained"])
    optimizer = utils.make_optimizer(t_cell.parameters(), config["optimizer"])
    print("frozen HCell, training TCell, lr {}".format(optimizer.param_groups[0]["lr"]), flush=True)

    if args.steps > 0:
        leaf(train_loader).set_epoch(0)
        limited = []
        for index, batch in enumerate(train_loader):
            if index >= args.steps:
                break
            limited.append(batch)
        class Once:
            def __iter__(self):
                return iter(limited)
            def __len__(self):
                return len(limited)
        loss = train_tcell.train(Once(), h_cell, t_cell, optimizer, 1)
        print("benchmark steps {} loss {:.4f}".format(len(limited), loss), flush=True)
        return

    epochs = config["epoch_max"]
    for epoch in range(1, epochs + 1):
        leaf(train_loader).set_epoch(epoch)
        loss = train_tcell.train(train_loader, h_cell, t_cell, optimizer, epoch)
        with torch.no_grad():
            val_psnr = train_tcell.valid(valid_loader, h_cell, t_cell)
        spec_t = dict(spec_t)
        spec_h = dict(spec_h)
        spec_t["sd"] = t_cell.state_dict()
        spec_h["sd"] = h_cell.state_dict()
        checkpoint = {
            "T_model": spec_t,
            "H_model": spec_h,
            "optimizer": {"name": config["optimizer"]["name"], "args": config["optimizer"]["args"], "sd": optimizer.state_dict()},
            "epoch": epoch,
        }
        torch.save(checkpoint, os.path.join(save_path, "epoch-last.pth"))
        if epoch % config.get("epoch_save", 5) == 0:
            torch.save(checkpoint, os.path.join(save_path, "epoch-{}.pth".format(epoch)))
        print("[{}/{}] loss {:.4f} val PSNR {:.4f}".format(epoch, epochs, loss, val_psnr), flush=True)


if __name__ == "__main__":
    main()
