"""Histogram-matched sparse-GT metrics for the driving stitches.

This is a local reimplementation of the RopStitch metric contract
(``eval_sparse_full.py``, ``eval_sparse_regions.py``, ``eval_hm_dataset.py``).
Nothing is imported from RopStitch.

Contract:
  * Images are RGB float64 in ``[0, 1]``. The mask is the sparse GT mask
    (``L > 127``). The prediction is histogram-matched to the GT on the masked
    pixels only via ``skimage.exposure.match_histograms``.
  * ``mae = mean(|matched - target|)``, ``mse = mean(diff**2)``,
    ``rmse = sqrt(mse)``.
  * ``ssim`` is the per-pixel luminance term
    ``mean((2*m*t + c1) / (m**2 + t**2 + c1))`` with ``c1 = 0.01**2``.
  * ``psnr = 10*log10(1/mse)`` (``inf`` when ``mse == 0``).
  * MAE/RMSE are reported on the 0-255 scale, PSNR on ``[0, 1]``. No LPIPS.
  * Regions are the three equal widths of the canvas; each third is
    histogram-matched independently. L+R and L+R+C are unweighted means of the
    region means, not a second pixel pool.
  * Dataset aggregation pools every frame; the PSNR mean skips non-finite
    values. Full-image and each region are separate pools.
"""
import argparse
import inspect
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.exposure import match_histograms

REGION_ORDER = ("Left", "Center", "Right")

# scikit-image renamed the multichannel flag to ``channel_axis`` in 0.19; the
# repository environment pins an older release. The result is identical, so use
# the modern call when available and fall back otherwise.
_HAS_CHANNEL_AXIS = "channel_axis" in inspect.signature(match_histograms).parameters


# ---------------------------------------------------------------------------
# Dataset registry (paths are relative to the data root)
# ---------------------------------------------------------------------------
DATASETS = {
    "nuscenes": {
        "images": "nuscenes/processed_10Hz_v2",
        "gt": "nuscenes/sparseWideFOVImages3_1554x294",
        "size": (1554, 294),
        "cam": 5,
        "val_list": "nuscenes/processed_10Hz_v2/nuScenes_Val.txt",
    },
    "lyft1920": {
        "images": "lyft/lyft_val1920_3cams",
        "gt": "lyft/1920_sparseWideFOVImages3_1554x294",
        "size": (1554, 294),
        "cam": 5,
        "val_list": "lyft/lyft_val1920_3cams/lyft_val1920.txt",
    },
    "lyft1224": {
        "images": "lyft/lyft_val1224_3cams",
        "gt": "lyft/1224_sparseWideFOVImages3_1554x434",
        "size": (1554, 434),
        "cam": 5,
        "val_list": "lyft/lyft_val1224_3cams/lyft_val1224.txt",
    },
    "widedrive3": {
        "images": "WideDrive3_processed/WideDriveVal",
        "gt": "WideDrive_processed/sparseWideFOVImages3_1554x294",
        "size": (1554, 294),
        "cam": 2,  # camera 2 is only the GT filename; stitching still uses 5/4/3
        "direct": True,  # dense virtual GT: no histogram matching
        "val_list": "WideDrive3_processed/WideDriveVal/val.txt",
    },
    "ddad": {
        "images": "ddad_process/valid",
        "gt": None,
        "size": None,
        "cam": 5,
        "val_list": None,
    },
}

LYFT_PARTS = ("lyft1920", "lyft1224")


def default_data_root():
    """Data root for the driving datasets, overridable with ``REWARP_DATA_ROOT``."""
    env = os.environ.get("REWARP_DATA_ROOT")
    if env:
        return Path(env)
    # <ImageStitch>/RopStitch_infer/datasets, sibling of this repository.
    return Path(__file__).resolve().parents[2] / "RopStitch_infer" / "datasets"


def has_gt(dataset):
    spec = DATASETS.get(dataset)
    return bool(spec and spec.get("gt"))


# ---------------------------------------------------------------------------
# Core metric math
# ---------------------------------------------------------------------------
def frame_hm(render, gt, mask, match=True):
    """Histogram-matched scores, or direct scores when match is false."""
    if mask.sum() == 0:
        return None
    pred = render[mask]
    target = gt[mask]
    if not match:
        matched = pred
        diff = matched - target
        mae = float(np.mean(np.abs(diff)))
        mse = float(np.mean(diff ** 2))
        c1 = 0.01 ** 2
        ssim = float(np.mean((2 * matched * target + c1) / (matched ** 2 + target ** 2 + c1)))
        return {
            "hm_mae": mae,
            "hm_rmse": float(np.sqrt(mse)),
            "hm_psnr": float(10.0 * np.log10(1.0 / mse)) if mse > 0 else float("inf"),
            "hm_ssim": ssim,
        }
    if _HAS_CHANNEL_AXIS:
        matched = match_histograms(pred[np.newaxis, ...], target[np.newaxis, ...], channel_axis=-1)[0]
    else:
        matched = match_histograms(pred[np.newaxis, ...], target[np.newaxis, ...], multichannel=True)[0]
    diff = matched - target
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff ** 2))
    c1 = 0.01 ** 2
    ssim = float(np.mean((2 * matched * target + c1) / (matched ** 2 + target ** 2 + c1)))
    return {
        "hm_mae": mae,
        "hm_rmse": float(np.sqrt(mse)),
        "hm_psnr": float(10.0 * np.log10(1.0 / mse)) if mse > 0 else float("inf"),
        "hm_ssim": ssim,
    }


def find_gt(gt_dir, frame, cam):
    """Locate ``{gt_dir}/rgb/{frame}_{cam}_sparse_wide.png`` and its mask."""
    suffix = f"_{cam}_sparse_wide.png"
    rgb = os.path.join(str(gt_dir), "rgb", f"{frame}{suffix}")
    mask = os.path.join(str(gt_dir), "mask", f"{frame}{suffix}")
    if os.path.exists(rgb) and os.path.exists(mask):
        return rgb, mask
    return None, None


def match_gt(render, gt, mask):
    """Resize GT (bilinear) and mask (nearest) to the render size when needed."""
    if gt.shape[:2] == render.shape[:2] and mask.shape == render.shape[:2]:
        return gt, mask
    size = (render.shape[1], render.shape[0])
    gt_img = Image.fromarray(np.clip(gt * 255.0, 0, 255).astype(np.uint8)).resize(size, Image.BILINEAR)
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255)).resize(size, Image.NEAREST)
    return np.array(gt_img).astype(np.float64) / 255.0, np.array(mask_img) > 127


def region_slices(width):
    """Three equal-width column slices. Width not divisible by 3 is an error."""
    if width % 3 != 0:
        raise SystemExit(f"width {width} is not divisible by 3")
    orig_w = width // 3
    return {
        "Left": (0, orig_w),
        "Center": (orig_w, orig_w * 2),
        "Right": (orig_w * 2, orig_w * 3),
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def mean_rows(rows):
    """Mean over frames. The PSNR mean skips non-finite values."""
    if not rows:
        return None
    return {
        "hm_psnr": float(np.mean([row["hm_psnr"] for row in rows if np.isfinite(row["hm_psnr"])])),
        "hm_ssim": float(np.mean([row["hm_ssim"] for row in rows])),
        "hm_mae": float(np.mean([row["hm_mae"] for row in rows])),
        "hm_rmse": float(np.mean([row["hm_rmse"] for row in rows])),
        "n": len(rows),
    }


def mean_of(stats):
    """Unweighted mean of per-region means."""
    keys = ("hm_psnr", "hm_ssim", "hm_mae", "hm_rmse")
    return {key: float(np.mean([item[key] for item in stats])) for key in keys}


def format_stats(stats, frames=True):
    text = (
        f"HM-PSNR: {stats['hm_psnr']:.2f}, HM-SSIM: {stats['hm_ssim']:.4f}, "
        f"HM-MAE: {stats['hm_mae'] * 255:.2f}, HM-RMSE: {stats['hm_rmse'] * 255:.2f}"
    )
    if frames and "n" in stats:
        text += f", Frames: {stats['n']}"
    return text


def score_spec(spec, pred_root, data_root):
    """Score every ``{scene}/{frame}.png`` under ``pred_root`` for one dataset."""
    pred_root = Path(pred_root)
    if not pred_root.is_dir():
        raise SystemExit(f"no predictions under {pred_root}")
    scenes = sorted(name for name in os.listdir(pred_root) if os.path.isdir(pred_root / name))
    full_rows = []
    region_rows = {name: [] for name in REGION_ORDER}
    per_scene = []
    skipped = 0
    for scene in scenes:
        pred_dir = pred_root / scene
        gt_dir = Path(data_root) / spec["gt"] / scene
        names = sorted(name for name in os.listdir(pred_dir) if name.lower().endswith(".png"))
        scene_full = []
        scene_regions = {name: [] for name in REGION_ORDER}
        for name in names:
            frame = os.path.splitext(name)[0]
            gt_path, mask_path = find_gt(gt_dir, frame, spec["cam"])
            if gt_path is None:
                skipped += 1
                continue
            render = np.array(Image.open(pred_dir / name).convert("RGB")).astype(np.float64) / 255.0
            gt = np.array(Image.open(gt_path).convert("RGB")).astype(np.float64) / 255.0
            mask = np.array(Image.open(mask_path).convert("L")) > 127
            gt, mask = match_gt(render, gt, mask)
            direct = spec.get("direct")
            if direct:
                mask = np.ones(render.shape[:2], dtype=bool)
            row = frame_hm(render, gt, mask, match=not direct)
            if row is None:
                skipped += 1
                continue
            row["frame"] = frame
            scene_full.append(row)
            slices = region_slices(render.shape[1])
            for region, (x0, x1) in slices.items():
                region_row = frame_hm(render[:, x0:x1], gt[:, x0:x1], mask[:, x0:x1], match=not spec.get("direct"))
                if region_row is not None:
                    region_row["frame"] = frame
                    scene_regions[region].append(region_row)
        if scene_full:
            full_rows.extend(scene_full)
            for region in REGION_ORDER:
                region_rows[region].extend(scene_regions[region])
            per_scene.append(
                (scene, mean_rows(scene_full), {name: mean_rows(rows) for name, rows in scene_regions.items()})
            )
            print(f"  {scene}: {len(scene_full)} frames", flush=True)
    return full_rows, region_rows, per_scene, skipped


def report_lines(title, full_rows, region_rows, per_scene, skipped, extra):
    full_stats = mean_rows(full_rows)
    region_stats = {name: mean_rows(region_rows[name]) for name in REGION_ORDER}
    lines = [
        f"=== {title} pairwise HM ===",
        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        *extra,
        "Mask: sparse GT mask only. LPIPS is not computed.",
        "HM-SSIM is the per-pixel luminance term.",
        "L+R and L+R+C are unweighted means of the region means.",
        "MAE and RMSE are on the 0-255 scale. PSNR uses [0, 1] images.",
        f"Skipped frames: {skipped}.",
        "",
        "Full",
        format_stats(full_stats) if full_stats else "no frames",
        "",
        "Regions",
    ]
    for name in REGION_ORDER:
        stats = region_stats[name]
        lines.append(f"{name:8s}: {format_stats(stats) if stats else 'no frames'}")
    if all(region_stats[name] for name in ("Left", "Right")):
        lines.append("L+R    : " + format_stats(mean_of([region_stats["Left"], region_stats["Right"]]), frames=False))
    if all(region_stats.values()):
        lines.append("L+R+C  : " + format_stats(mean_of(list(region_stats.values())), frames=False))
    lines.append("")
    lines.append("Per-scene Full")
    for scene, full_stats, _regions in per_scene:
        lines.append(f"{scene}: {format_stats(full_stats)}")
    return lines


def write_report(path, lines):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    with open(path, "w") as handle:
        handle.write(text)
    print(text, end="")
    print(f"Wrote {path}", flush=True)


def pred_root_for(output_root, dataset):
    return Path(output_root) / "metric_eval" / dataset / "rewarp"


def default_report_path(output_root, dataset):
    return Path(output_root) / "metric_eval" / dataset / "rewarp_hm.txt"


def run(dataset, output_root="output", data_root=None, report_path=None):
    """Score one dataset (or pooled ``lyft``) and write the HM report."""
    if dataset == "ddad":
        raise SystemExit("ddad has no sparse GT; eval_hm does not support ddad")
    data_root = Path(data_root) if data_root is not None else default_data_root()

    if dataset == "lyft":
        parts = []
        for name in LYFT_PARTS:
            print(f"scoring {name}", flush=True)
            pred_root = pred_root_for(output_root, name)
            parts.append((name, score_spec(DATASETS[name], pred_root, data_root)))
        full_rows = [row for _name, (rows, _regions, _scenes, _skip) in parts for row in rows]
        if not full_rows:
            raise SystemExit(f"no scored frames for lyft under {output_root}/metric_eval")
        region_rows = {name: [] for name in REGION_ORDER}
        per_scene = []
        skipped = 0
        extra = ["Pool: every lyft1920 and lyft1224 frame. Subsets are not averaged."]
        for name, (rows, regions, scenes, skip) in parts:
            skipped += skip
            for region in REGION_ORDER:
                region_rows[region].extend(regions[region])
            stats = mean_rows(rows)
            extra.append(f"{name}: {format_stats(stats) if stats else 'no frames'}")
            per_scene.extend(
                (f"{name}/{scene}", full_stats, region_stats) for scene, full_stats, region_stats in scenes
            )
        path = report_path or default_report_path(output_root, "lyft")
        write_report(path, report_lines("lyft", full_rows, region_rows, per_scene, skipped, extra))
        return path

    spec = DATASETS[dataset]
    print(f"scoring {dataset}", flush=True)
    pred_root = pred_root_for(output_root, dataset)
    full_rows, region_rows, per_scene, skipped = score_spec(spec, pred_root, data_root)
    if not full_rows:
        raise SystemExit(f"no scored frames under {pred_root}")
    extra = [f"Pred: {pred_root}", f"GT: {data_root / spec['gt']}", f"GT camera: {spec['cam']}"]
    path = report_path or default_report_path(output_root, dataset)
    write_report(path, report_lines(dataset, full_rows, region_rows, per_scene, skipped, extra))
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Dataset-level sparse-GT HM metrics for the driving stitches")
    parser.add_argument("--dataset", required=True, choices=[*DATASETS, "lyft"])
    parser.add_argument("--output", default="output", help="output root holding metric_eval/")
    parser.add_argument("--data-root", default=None, help="dataset root (defaults to the local layout)")
    parser.add_argument("--report", default=None, help="explicit report path")
    args = parser.parse_args(argv)
    run(args.dataset, args.output, args.data_root, args.report)


if __name__ == "__main__":
    main()
