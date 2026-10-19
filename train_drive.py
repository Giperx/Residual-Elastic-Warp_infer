#!/usr/bin/env python
"""Fine-tune REwarp on driving camera pairs, in the paper's order.

Stage hcell updates HCell. Stage tcell freezes HCell and updates TCell.
Validation is not run. `train.py` runs hcell and then tcell.

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
import train_hcell
import train_tcell
from rewarp_drive.compat import install as install_gesv

install_gesv()


def leaf(loader):
    dataset = loader.dataset
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    return dataset


def make_loader(spec, tag, workers):
    dataset = datasets.make(spec["dataset"])
    dataset = datasets.make(spec["wrapper"], args={"dataset": dataset})
    print("{} dataset: size={} pool={}".format(tag, len(dataset), len(leaf_of(dataset).pairs)), flush=True)
    loader = DataLoader(
        dataset,
        batch_size=spec["batch_size"],
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    return loader


def leaf_of(dataset):
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    return dataset


def load_cells(path, stage):
    checkpoint = torch.load(path, map_location="cpu")
    h_cell = models.make(checkpoint["H_model"], load_sd=True)
    t_cell = models.make(checkpoint["T_model"], load_sd=True)
    if stage == "hcell":
        h_cell.cuda().train()
        for parameter in h_cell.parameters():
            parameter.requires_grad = True
        for parameter in t_cell.parameters():
            parameter.requires_grad = False
    else:
        h_cell.cuda().eval()
        t_cell.cuda().train()
        for parameter in h_cell.parameters():
            parameter.requires_grad = False
    return h_cell, t_cell, checkpoint["H_model"], checkpoint["T_model"]


def save_checkpoint(path, spec_h, spec_t, h_cell, t_cell, optimizer, config, epoch):
    spec_h = dict(spec_h)
    spec_t = dict(spec_t)
    spec_h["sd"] = h_cell.state_dict()
    spec_t["sd"] = t_cell.state_dict()
    checkpoint = {
        "T_model": spec_t,
        "H_model": spec_h,
        "optimizer": {
            "name": config["optimizer"]["name"],
            "args": config["optimizer"]["args"],
            "sd": optimizer.state_dict(),
        },
        "epoch": epoch,
    }
    torch.save(checkpoint, path)
    print("saved {}".format(path), flush=True)


def main():
    parser = ArgumentParser(description="Fine-tune REwarp on cam5+cam4 and cam5+cam3")
    parser.add_argument("--config", default="configs/train/drive_tcell.yaml")
    parser.add_argument("--dataset", default=None, choices=["nuscenes", "lyft", "widedrive3"])
    parser.add_argument("--stage", choices=("hcell", "tcell"), default="tcell")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--samples", type=int, default=None, help="random pairs per epoch")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--steps", type=int, default=0, help="run this many steps and stop; for timing")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    with open(args.config) as handle:
        config = yaml.load(handle, Loader=yaml.FullLoader)
    name = args.dataset or config.get("dataset", "nuscenes")
    defaults = {"nuscenes": 1, "lyft": 3, "widedrive3": 3}
    spec = config["train_dataset"]
    spec["dataset"]["args"]["dataset"] = name
    if args.samples:
        spec["dataset"]["args"]["samples_per_epoch"] = args.samples
    if args.batch_size:
        spec["batch_size"] = args.batch_size
    elif args.stage == "hcell":
        spec["batch_size"] = 16
    config["epoch_max"] = args.epochs if args.epochs else defaults.get(name, config["epoch_max"])
    if args.pretrained:
        config["pretrained"] = args.pretrained

    save_path = os.path.join("save", "drive_" + name, args.stage)
    os.makedirs(save_path, exist_ok=True)
    train_hcell.config = config
    train_tcell.config = config
    train_tcell.log = print

    train_loader = make_loader(config["train_dataset"], "train", args.workers)
    h_cell, t_cell, spec_h, spec_t = load_cells(config["pretrained"], args.stage)
    trainable = h_cell.parameters() if args.stage == "hcell" else t_cell.parameters()
    optimizer = utils.make_optimizer(trainable, config["optimizer"])
    print(
        "{} stage, lr {}".format(args.stage, optimizer.param_groups[0]["lr"]),
        flush=True,
    )

    def run_epoch(loader, epoch):
        if args.stage == "hcell":
            return train_hcell.train(loader, h_cell, optimizer, epoch)
        return train_tcell.train(loader, h_cell, t_cell, optimizer, epoch)

    def write(epoch):
        path = os.path.join(save_path, "epoch-last.pth")
        save_checkpoint(path, spec_h, spec_t, h_cell, t_cell, optimizer, config, epoch)
        if epoch % config.get("epoch_save", 5) == 0:
            save_checkpoint(
                os.path.join(save_path, "epoch-{}.pth".format(epoch)),
                spec_h, spec_t, h_cell, t_cell, optimizer, config, epoch,
            )

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

        loss = run_epoch(Once(), 1)
        print("benchmark steps {} loss {:.4f}".format(len(limited), loss), flush=True)
        write(1)
        return

    epochs = config["epoch_max"]
    for epoch in range(1, epochs + 1):
        leaf(train_loader).set_epoch(epoch)
        loss = run_epoch(train_loader, epoch)
        write(epoch)
        print("[{}/{}] {} loss {:.4f}".format(epoch, epochs, args.stage, loss), flush=True)


if __name__ == "__main__":
    main()
