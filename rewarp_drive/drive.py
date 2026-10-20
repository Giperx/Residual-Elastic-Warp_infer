"""Stitch driving cameras 5+4 and 5+3 for one frame or a validation list.

Camera 5 is the reference. The two pairwise stitches are composited so their
camera-5 rectangles coincide. A finished frame is written twice, in order:

1. ``output/stitch/{dataset}/{scene}/{stem}.jpg`` — full stitch, for viewing.
2. ``output/metric_eval/{dataset}/rewarp/{scene}/{stem}.png`` — the camera-5
   crop resized to the sparse GT. This is the only image ``eval_hm`` scores.

Masks and the unresized crop are not written.

Failed frames write neither file, so they stay out of the metric mean.
Existing pairs are left in place unless ``--overwrite`` is set.

``--gpu`` must take effect before torch is imported, so the GPU environment
variable is parsed from ``sys.argv`` at import time, mirroring the remote
driver. Run it through the repository entry point::

    python stitch_drive.py --dataset nuscenes --scene 002 --frame 0
"""
import os
from argparse import ArgumentParser
from pathlib import Path


def _preparse_gpu():
    """Parse ``--gpu`` from argv before torch is imported."""
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--gpu", default="0")
    known, _ = parser.parse_known_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = known.gpu


_preparse_gpu()

import torch  # noqa: E402  (import after CUDA_VISIBLE_DEVICES is set)

from . import metrics  # noqa: E402
from . import stitch_core  # noqa: E402

IMAGE_EXTS = (".jpg", ".png", ".jpeg")


def frame_candidates(frame):
    """Filename stems to try for one frame.

    ``--frame 0`` and the on-disk stem ``000`` must both resolve. Numeric ids
    are tried at their given width and at 3 and 4 digits. Non-numeric stems
    (if a dataset ever uses them) are used unchanged.
    """
    text = str(frame).strip()
    names = []
    if text.isdigit():
        number = int(text)
        for width in (len(text), 3, 4):
            names.append(f"{number:0{width}d}")
        names.append(str(number))
    else:
        names.append(text)
    seen = []
    for name in names:
        if name not in seen:
            seen.append(name)
    return seen


def stem_name(frame):
    """Zero-pad a numeric frame id to at least three digits."""
    text = str(frame).strip()
    if text.isdigit():
        return f"{int(text):0{max(3, len(text))}d}"
    return text


def _as_int(value):
    return int(value.item()) if torch.is_tensor(value) else int(value)


def find_image(scene, frame, cam):
    """Find ``{stem}_{cam}.{jpg,png,jpeg}`` under ``{scene}/images``.

    Returns ``(path, stem)``. The stem is the one that exists on disk so the
    metric png name matches the sparse GT (``000`` rather than ``0``).
    """
    image_dir = Path(scene) / "images"
    tried = []
    for stem in frame_candidates(frame):
        for ext in IMAGE_EXTS:
            path = image_dir / f"{stem}_{cam}{ext}"
            tried.append(path.name)
            if path.exists():
                return path, stem
    raise FileNotFoundError(f"missing camera {cam} frame {frame} under {image_dir} (tried {', '.join(tried)})")


def list_frames(scene):
    """Frame stems that have a camera-5 image, sorted."""
    image_dir = Path(scene) / "images"
    if not image_dir.is_dir():
        return []
    stems = set()
    for ext in IMAGE_EXTS:
        for path in image_dir.glob(f"*_5{ext}"):
            stems.add(path.name[: -len(f"_5{ext}")])
    return sorted(stems)


def visualization_path(output_root, dataset, scene_id, stem):
    """Full stitch kept for later viewing. Not read by the metric."""
    return Path(output_root) / "stitch" / dataset / scene_id / f"{stem}.jpg"


def metric_image_path(output_root, dataset, scene_id, stem):
    """GT-sized crop. This is the only image the metric scores."""
    return metrics.pred_root_for(output_root, dataset) / scene_id / f"{stem}.png"


def stitch_frame(
    dataset,
    scene_id,
    frame,
    data_root,
    output_root,
    checkpoint=None,
    device=None,
    models=None,
    save_pairs=False,
    save_chain=False,
    overwrite=False,
):
    """Stitch one frame. Write the full view image first, then the metric crop.

    A failed pair writes nothing, so ``eval_hm`` never sees that frame.
    """
    if dataset not in metrics.DATASETS:
        raise ValueError(f"dataset must be one of {', '.join(metrics.DATASETS)}")
    spec = metrics.DATASETS[dataset]
    scene = Path(data_root) / spec["images"] / scene_id
    if not scene.is_dir():
        raise FileNotFoundError(f"scene not found: {scene}")

    found = [find_image(scene, frame, cam) for cam in stitch_core.CAMERAS]
    paths = [item[0] for item in found]
    stem = found[0][1]
    view_path = visualization_path(output_root, dataset, scene_id, stem)
    if metrics.has_gt(dataset):
        crop_path = metric_image_path(output_root, dataset, scene_id, stem)
    else:
        crop_path = view_path.with_name(f"{stem}_crop.jpg")
    if not overwrite and view_path.exists() and crop_path.exists():
        print(f"keep {view_path.name}: stitch and crop already saved", flush=True)
        return True

    images = [stitch_core.process_image(str(path)) for path in paths]
    input_size = spec.get("input_size")
    if input_size:
        in_w, in_h = input_size
        images = [stitch_core.resize_input(image, in_w, in_h) for image in images]
    ref_h, ref_w = images[0].shape[-2:]

    if models is None:
        model_h, model_t = stitch_core.load_models(checkpoint, device)
    else:
        model_h, model_t = models
    stitch_core.MAX_CANVAS_SIZE = 10 ** 9
    device = device if device is not None else stitch_core.DEVICE

    with torch.no_grad():
        try:
            first, _, offset_1 = stitch_core.stitch_blended(images[0], images[1], model_h, model_t)
        except stitch_core.StitchingError as error:
            print(f"{dataset} {scene_id} frame {stem}: cameras 5+4 failed: {error}")
            return False
        origin_54 = (_as_int(offset_1[0]), _as_int(offset_1[1]))
        print(f"cameras 5+4 side pixels L/R {stitch_core.side_counts(first, origin_54, ref_w)}", flush=True)
        pair_54 = first.detach().cpu()
        del first
        torch.cuda.empty_cache()

        # Optional. The metric uses the pairwise composite, not this chain.
        if save_chain:
            try:
                chained, _, offset_chain = stitch_core.stitch_blended(pair_54, images[2], model_h, model_t)
                chain_origin = (
                    origin_54[0] + _as_int(offset_chain[0]),
                    origin_54[1] + _as_int(offset_chain[1]),
                )
                chain_path = Path(output_root) / "stitch" / dataset / scene_id / "chain" / f"{stem}.jpg"
                stitch_core.save_picture((chained + 1) / 2, chain_path)
                print(f"saved {chain_path} {tuple(chained.shape[-2:])}", flush=True)
                del chained
                torch.cuda.empty_cache()
            except Exception as error:
                print(f"{dataset} {scene_id} frame {stem}: chained 5+4 then 3 failed: {error}")
                torch.cuda.empty_cache()

        try:
            pair_53, _, offset_53 = stitch_core.stitch_blended(images[0], images[2], model_h, model_t)
        except stitch_core.StitchingError as error:
            print(f"{dataset} {scene_id} frame {stem}: cameras 5+3 failed: {error}")
            return False
        origin_53 = (_as_int(offset_53[0]), _as_int(offset_53[1]))
        print(f"cameras 5+3 side pixels L/R {stitch_core.side_counts(pair_53, origin_53, ref_w)}", flush=True)
        final, origin = stitch_core.composite_by_reference(pair_54, origin_54, pair_53.detach().cpu(), origin_53)

    # JPEG cannot store a side longer than 65500. Skip the frame instead of aborting the dataset.
    if max(final.shape[-2:]) > 65500:
        print(
            f"{dataset} {scene_id} frame {stem}: canvas {tuple(final.shape[-2:])} exceeds JPEG limit, skip",
            flush=True,
        )
        return False

    # View image first, then the crop that eval_hm reads. Nothing is written before this.
    try:
        stitch_core.save_picture((final + 1) / 2, view_path)
    except OSError as error:
        if view_path.exists():
            view_path.unlink()
        print(f"{dataset} {scene_id} frame {stem}: could not save the stitch ({error}), skip", flush=True)
        return False
    print(
        f"saved view {view_path} {tuple(final.shape[-2:])} "
        f"side pixels L/R {stitch_core.side_counts(final, origin, ref_w)}",
        flush=True,
    )
    if metrics.has_gt(dataset):
        gt_w, gt_h = spec["size"]
        shape = stitch_core.save_metric_image(final, origin, ref_h, ref_w, gt_w, gt_h, None, crop_path)
    else:
        shape = stitch_core.save_metric_image(final, origin, ref_h, ref_w, None, None, crop_path, None)
    print(f"saved crop {crop_path} {shape[1]}x{shape[0]}", flush=True)
    if save_pairs:
        pair_dir = Path(output_root) / "stitch" / dataset / scene_id / "pairs"
        stitch_core.save_picture((pair_54 + 1) / 2, pair_dir / f"{stem}_54.jpg")
        stitch_core.save_picture((pair_53 + 1) / 2, pair_dir / f"{stem}_53.jpg")
        print(f"saved pairs under {pair_dir}", flush=True)
    return True


def read_list(path):
    with open(path) as handle:
        return [line.strip() for line in handle if line.strip()]


def resolve_scenes(dataset, scene, val_list, data_root, limit):
    if scene:
        scenes = [scene]
    else:
        list_path = val_list or metrics.DATASETS[dataset].get("val_list")
        if not list_path:
            raise SystemExit("pass --scene or --val-list")
        list_path = Path(list_path)
        if not list_path.is_absolute() and not list_path.exists():
            list_path = Path(data_root) / list_path
        scenes = read_list(list_path)
    if limit and limit > 0:
        scenes = scenes[:limit]
    return scenes


def main(argv=None):
    parser = ArgumentParser(description="Stitch driving cameras 5+4 and 5+3 (camera 5 reference)")
    parser.add_argument("--dataset", required=True, choices=sorted(metrics.DATASETS))
    parser.add_argument("--scene", default=None)
    parser.add_argument("--frame", default=None, help="frame stem or frame index")
    parser.add_argument("--val-list", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output", default="output")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--checkpoint", default=str(stitch_core.CHECKPOINT))
    parser.add_argument("--save-pairs", action="store_true", help="also save the 5+4 and 5+3 images")
    parser.add_argument("--chain", action="store_true", help="also stitch 5+4 then 3; not used for metrics")
    parser.add_argument("--overwrite", action="store_true", help="rewrite a frame that already has a view and a crop")
    args = parser.parse_args(argv)

    # Kept for parity with the remote driver; the effective value was set at import.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    data_root = Path(args.data_root) if args.data_root else metrics.default_data_root()
    scenes = resolve_scenes(args.dataset, args.scene, args.val_list, data_root, args.limit)
    if not scenes:
        raise SystemExit("no scenes to stitch")

    device = stitch_core.DEVICE
    print(f"{args.dataset}: {len(scenes)} scenes on {device} -> {args.output}", flush=True)
    models = stitch_core.load_models(args.checkpoint, device)
    saved = failed = 0
    for scene_index, scene_id in enumerate(scenes):
        scene = Path(data_root) / metrics.DATASETS[args.dataset]["images"] / scene_id
        frames = [args.frame] if args.frame is not None else list_frames(scene)
        if not frames:
            print(f"scene {scene_index + 1}/{len(scenes)} {scene_id}: no frames", flush=True)
            continue
        print(f"scene {scene_index + 1}/{len(scenes)} {scene_id}: {len(frames)} frames", flush=True)
        for frame in frames:
            try:
                ok = stitch_frame(
                    args.dataset,
                    scene_id,
                    frame,
                    data_root,
                    args.output,
                    args.checkpoint,
                    device,
                    models=models,
                    save_pairs=args.save_pairs,
                    save_chain=args.chain,
                    overwrite=args.overwrite,
                )
            except (FileNotFoundError, stitch_core.StitchingError, RuntimeError, OSError) as error:
                print(f"  skip {scene_id} {stem_name(frame)}: {error}", flush=True)
                torch.cuda.empty_cache()
                failed += 1
                continue
            if ok:
                saved += 1
            else:
                failed += 1
    print(f"done: saved {saved}, failed {failed}", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
