# Residual Elastic Warps
This repository contains the official implementation of [REwarp](https://arxiv.org/abs/2309.01406), 24' WACV.

## Requirement
1) Python packages
```
conda env create --file environment.yaml
conda activate rewarp
```

pip install torchgeometry==0.1.2 torchmetrics==0.8.2 -i https://pypi.tuna.tsinghua.edu.cn/simple

20260123comment:
conda activate rewarp 
2024 WACV Learning Residual Elastic Warps for Image Stitching under Dirichlet Boundary Condition

## Train & Evaluation
```
bash scripts/train.sh 0
bash scripts/eval.sh 0
```

## Pretrained Models
You can download below models on this [link](https://drive.google.com/file/d/1T4G2qDTwvSWCPyxx7Q-tM0F64qwvvGke/view?usp=share_link).
1. hcell.pth: Pretrained HCell,
2. rewarp.pth: Pretrained HCell and TCell.

## Acknowlegment
This work is mainly based on [LTEW](https://github.com/jaewon-lee-b/ltew) and [IHN](https://github.com/imdumpl78/IHN), we thank the authors for their contributions.

## Driving fine-tune

TCell can be fine-tuned on camera pairs from the driving training splits. Each sample is still two images: camera 5 with camera 4, or camera 5 with camera 3. HCell stays frozen. Every epoch draws a shuffled subset of scenes and frames (`samples_per_epoch`), resizes both images to 512, and uses the original overlap L1 loss.

Training starts from `pretrained/rewarp.pth`. One epoch is one shuffled pass over every camera pair. The defaults are 1 epoch for nuScenes, 3 epochs for lyft1920 and lyft1224 mixed together, and 3 epochs for WideDrive3.

```
python train_drive.py --dataset nuscenes --gpu 1
python train_drive.py --dataset lyft --gpu 1
python train_drive.py --dataset widedrive3 --gpu 1
```

WideDrive3 training reads `WideDrive3_processed/WideDriveTrain/train.txt`, the same layout as `WideDriveVal` but a sibling directory. Validation stays on `WideDriveVal/val.txt`. Checkpoints go to `save/drive_{dataset}/`.

## Driving datasets

This repository also ships the driving-scene stitch used by the sparse wide-FOV
evaluation. Cameras `5, 4, 3` are stitched with camera 5 as the reference: the
`5+4` and `5+3` pairwise panoramas are composited so their camera-5 rectangles
coincide. The chained `5+4 then 3` result is optional (`--chain`) and is not scored.

Datasets are read from `REWARP_DATA_ROOT` (default
`../RopStitch_infer/datasets`, override with `--data-root`). Supported sets:
`nuscenes`, `lyft1920`, `lyft1224`, `widedrive3` (with sparse GT) and `ddad`
(images only, stitch only).

Source cameras are resized before stitching, matching the compared model: `518x294` for nuScenes, lyft1920, and WideDrive3, and `518x434` for lyft1224. The metric crop is then three of those widths.

Stitch one frame (run inside the `rewarp` conda environment):

```
python stitch_drive.py --dataset nuscenes --scene 002 --frame 0
```

Each successful frame is saved as two images:

1. `output/stitch/{dataset}/{scene}/{stem}.jpg` — the full stitch, for later viewing.
2. `output/metric_eval/{dataset}/rewarp/{scene}/{stem}.png` — camera 5 at its original height and width, with one camera-5 width on the left and one on the right, then resized to the sparse GT size (`INTER_AREA` when shrinking, else `INTER_LINEAR`).

Masks and the unresized crop are not written. `eval_hm.py` reads only the resized png. A frame that fails to stitch writes neither file. A rerun skips a frame that already has both (`--overwrite` to redo it). `--save-pairs` also keeps the separate `5+4` and `5+3` images.

Score one dataset:

```
python eval_hm.py --dataset nuscenes
```

`--dataset lyft` pools `lyft1920` and `lyft1224` frame by frame. The report is
written to `output/metric_eval/{dataset}/rewarp_hm.txt`.

Scores match the RopStitch sparse-GT histogram-matched metrics: the prediction
is histogram-matched to the GT on the sparse GT mask only, then PSNR, SSIM
(per-pixel luminance term), MAE, and RMSE are reported for the full canvas and
for the equal-width `L`/`C`/`R` thirds. `L+R` and `L+R+C` are unweighted means
of the region means. MAE and RMSE are on the 0-255 scale, PSNR on `[0, 1]`, and
PSNR aggregation skips non-finite frames. There is no LPIPS.

