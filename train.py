#!/usr/bin/env python3
"""Run REwarp driving finetuning as HCell, then frozen-HCell TCell.

Each stage uses the driving epoch count: nuscenes 1, lyft 3, widedrive3 3.

    python train.py --dataset widedrive3 --gpu 4 --epochs 3
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run_stage(stage, args, pretrained):
    command = [
        sys.executable,
        "-u",
        str(ROOT / "train_drive.py"),
        "--dataset",
        args.dataset,
        "--stage",
        stage,
        "--gpu",
        args.gpu,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--workers",
        str(args.workers),
        "--pretrained",
        pretrained,
    ]
    if args.samples:
        command.extend(["--samples", str(args.samples)])
    if args.steps:
        command.extend(["--steps", str(args.steps)])
    print(f"\n########## {stage} ##########", flush=True)
    subprocess.run(command, check=True, cwd=ROOT)


def main():
    parser = argparse.ArgumentParser(description="REwarp HCell then TCell finetuning")
    parser.add_argument("--dataset", required=True, choices=("nuscenes", "lyft", "widedrive3"))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--stage", choices=("both", "hcell", "tcell"), default="both")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pretrained", default=str(ROOT / "pretrained" / "rewarp.pth"))
    parser.add_argument("--samples", type=int, default=0)
    parser.add_argument("--steps", type=int, default=0)
    args = parser.parse_args()

    hcell_last = ROOT / "save" / f"drive_{args.dataset}" / "hcell" / "epoch-last.pth"
    if args.stage in ("both", "hcell"):
        run_stage("hcell", args, args.pretrained)
    if args.stage in ("both", "tcell"):
        pretrained = str(hcell_last if hcell_last.is_file() else args.pretrained)
        if not hcell_last.is_file():
            print(f"no HCell checkpoint, TCell starts from {pretrained}", flush=True)
        run_stage("tcell", args, pretrained)


if __name__ == "__main__":
    main()
