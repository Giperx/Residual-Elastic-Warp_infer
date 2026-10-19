import argparse
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyiqa
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
# # 使用全部默认 8 个指标
# python BatchCalculateMetrics_iqa.py

# # 只用 3 个快速指标
# python BatchCalculateMetrics_iqa.py --metrics niqe,brisque,musiq

# # 自定义组合
# python BatchCalculateMetrics_iqa.py --metrics niqe,clipiqa,topiq_nr,qalign

DEFAULT_SCENE_LIST = "nuScenes_Val.txt"
DEFAULT_RENDER_ROOT = "logs/omnire_nuscenes/omnire"
DEFAULT_OUTPUT = "logs/omnire_nuscenes/omnire/nuscenes_cam0_cam5_iqa_metrics.txt"
DEFAULT_CAM0_SUFFIX = "fd2"
DEFAULT_CAM5_SUFFIX = "bd2"
DEFAULT_BATCH_SIZE = 16
TARGET_CAMS = [0, 5]

DEFAULT_METRICS = "niqe,brisque,musiq,clipiqa,topiq_nr,arniqa,liqe"
# qalign # export HF_ENDPOINT=https://hf-mirror.com

METRIC_DESCRIPTIONS = {
    "niqe": "NIQE: lower is better (Natural Image Quality Evaluator, deviation from natural scene statistics).",
    "brisque": "BRISQUE: lower is better (Blind/Referenceless Image Spatial Quality Evaluator, widely-used traditional NR metric).",
    "musiq": "MUSIQ: higher is better (Multi-Scale Image Quality Transformer, trained on KonIQ-10k).",
    "clipiqa": "CLIPIQA: higher is better (CLIP-based No-Reference Image Quality Assessment).",
    "topiq_nr": "TOPIQ_NR: higher is better (Top-Down Approch from Semantics to Distortions, no-reference version, ResNet50 backbone).",
    "arniqa": "ARNIQA: higher is better (Autoregressive No-Reference Image Quality Assessment, self-supervised features).",
    "qalign": "QALIGN: higher is better, range [1,5] (Unified metric based on large vision-language model, most powerful NR-IQA to date).",
    "liqe": "LIQE: higher is better (Language-Guided Blind Image Quality Evaluation, strong cross-dataset generalization).",
}


class MetricSummary:
    def __init__(self, metric_names: Sequence[str]) -> None:
        self._metric_names = list(metric_names)
        self._values: Dict[str, List[float]] = {name: [] for name in self._metric_names}
        self.total: int = 0
        self.missing: int = 0

    def update(self, values: Dict[str, float]) -> None:
        for name in self._metric_names:
            self._values[name].append(values[name])
        self.total += 1

    def mark_missing(self) -> None:
        self.missing += 1

    def mean(self) -> Dict[str, float]:
        result: Dict[str, float] = {}
        for name in self._metric_names:
            vals = self._values[name]
            result[name] = float(np.mean(vals)) if vals else 0.0
        result["count"] = len(self._values[self._metric_names[0]]) if self._metric_names else 0
        result["missing"] = self.missing
        return result

    @property
    def metric_names(self) -> List[str]:
        return list(self._metric_names)


def load_scene_list(txt_path: str) -> List[str]:
    if not os.path.exists(txt_path):
        return []
    with open(txt_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def find_existing_image_path(folder: Path, stem: str) -> Optional[Path]:
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = folder / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def load_rgb_tensor(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    return transforms.ToTensor()(image)


def make_scene_stats(metric_names: Sequence[str]) -> Dict[str, MetricSummary]:
    return {
        "all": MetricSummary(metric_names),
        "cam0": MetricSummary(metric_names),
        "cam5": MetricSummary(metric_names),
    }


def format_summary(name: str, summary: MetricSummary) -> str:
    stats = summary.mean()
    parts = [f"{name:<8}", f"N={stats['count']:5d}", f"Missing={stats['missing']:5d}"]
    for metric_name in summary.metric_names:
        parts.append(f"{metric_name.upper()}={stats[metric_name]:.4f}")
    return " | ".join(parts)


def build_tasks(
    render_root: Path,
    scene_ids: Sequence[str],
    cam0_suffix: str,
    cam5_suffix: str,
) -> List[Dict[str, object]]:
    tasks: List[Dict[str, object]] = []

    cam_config = {
        0: cam0_suffix,
        5: cam5_suffix,
    }

    for scene_id in scene_ids:
        scene_pred_dirs = {
            cam_idx: render_root / f"3cams{scene_id}{suffix}" / "pseudoGTWide"
            for cam_idx, suffix in cam_config.items()
        }

        for cam_idx in TARGET_CAMS:
            pred_dir = scene_pred_dirs[cam_idx]
            if not pred_dir.exists():
                continue

            for file_name in sorted(os.listdir(pred_dir)):
                if not file_name.endswith((".png", ".jpg", ".jpeg")):
                    continue
                stem = Path(file_name).stem
                parts = stem.rsplit("_", 1)
                if len(parts) != 2:
                    continue
                frame_id = parts[0]
                pred_path = pred_dir / file_name
                tasks.append(
                    {
                        "scene_id": scene_id,
                        "frame_id": frame_id,
                        "cam_idx": cam_idx,
                        "pred_path": pred_path,
                    }
                )

    return tasks


def evaluate_tasks(
    tasks: Sequence[Dict[str, object]],
    metric_names: Sequence[str],
    device: torch.device,
    batch_size: int,
) -> Tuple[Dict[str, MetricSummary], Dict[str, Dict[str, MetricSummary]]]:
    metric_fns: Dict[str, object] = {}
    for name in metric_names:
        metric_fns[name] = pyiqa.create_metric(name, device=device)
        if hasattr(metric_fns[name], "eval") and callable(getattr(metric_fns[name], "eval")):
            metric_fns[name].eval()

    global_stats = make_scene_stats(metric_names)
    scene_stats: Dict[str, Dict[str, MetricSummary]] = defaultdict(lambda: make_scene_stats(metric_names))

    sorted_indices: List[int] = sorted(
        range(len(tasks)),
        key=lambda i: (
            tasks[i]["scene_id"],
            tasks[i]["cam_idx"],
            tasks[i]["frame_id"],
        ),
    )
    sorted_tasks = [tasks[i] for i in sorted_indices]

    for start_idx in tqdm(range(0, len(sorted_tasks), batch_size), desc="Evaluating"):
        batch_tasks = sorted_tasks[start_idx : start_idx + batch_size]

        valid_tasks: List[Dict[str, object]] = []
        pred_list: List[torch.Tensor] = []

        for task in batch_tasks:
            scene_id = task["scene_id"]
            cam_idx = task["cam_idx"]
            cam_key = "cam0" if cam_idx == 0 else "cam5"

            try:
                pred_tensor = load_rgb_tensor(task["pred_path"])
            except Exception:
                global_stats[cam_key].mark_missing()
                global_stats["all"].mark_missing()
                scene_stats[scene_id][cam_key].mark_missing()
                scene_stats[scene_id]["all"].mark_missing()
                continue

            valid_tasks.append(task)
            pred_list.append(pred_tensor)

        if not valid_tasks:
            continue

        size_groups: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for idx, pred_tensor in enumerate(pred_list):
            h, w = pred_tensor.shape[-2], pred_tensor.shape[-1]
            size_groups[(h, w)].append(idx)

        for (h, w), group_indices in size_groups.items():
            group_tasks = [valid_tasks[i] for i in group_indices]
            group_preds = [pred_list[i] for i in group_indices]
            pred_batch = torch.stack(group_preds).to(device)

            metric_batches: Dict[str, torch.Tensor] = {}
            with torch.inference_mode():
                for name, fn in metric_fns.items():
                    if name == "qalign":
                        scores = [torch.atleast_1d(fn(pred_batch[i:i+1])).item() for i in range(pred_batch.shape[0])]
                        batch_scores = torch.tensor(scores, dtype=torch.float32)
                    else:
                        batch_scores = torch.atleast_1d(fn(pred_batch)).detach().cpu().reshape(-1)
                    assert batch_scores.shape[0] == len(group_tasks), (
                        f"{name} returned {batch_scores.shape[0]} scores for {len(group_tasks)} images"
                    )
                    metric_batches[name] = batch_scores

            for idx, task in enumerate(group_tasks):
                scene_id = task["scene_id"]
                cam_idx = task["cam_idx"]
                cam_key = "cam0" if cam_idx == 0 else "cam5"

                values = {name: metric_batches[name][idx].item() for name in metric_names}

                global_stats[cam_key].update(values)
                global_stats["all"].update(values)
                scene_stats[scene_id][cam_key].update(values)
                scene_stats[scene_id]["all"].update(values)

    return global_stats, scene_stats


def write_report(
    output_path: Path,
    scene_ids: Sequence[str],
    tasks: Sequence[Dict[str, object]],
    global_stats: Dict[str, MetricSummary],
    scene_stats: Dict[str, Dict[str, MetricSummary]],
    render_root: Path,
    scene_list_path: Path,
    cam0_suffix: str,
    cam5_suffix: str,
    batch_size: int,
    metric_names: Sequence[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("NuScenes Cam0/Cam5 No-Reference IQA Metric Report\n")
        f.write(f"Scene list: {scene_list_path}\n")
        f.write(f"Render root: {render_root}\n")
        f.write(f"Cam0 run suffix: {cam0_suffix}\n")
        f.write(f"Cam5 run suffix: {cam5_suffix}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Metrics: {', '.join(metric_names)}\n")
        f.write(f"Tasks collected: {len(tasks)}\n")
        f.write("No-reference metrics computed on full rendered images (no center crop, no ego mask).\n")
        for name in metric_names:
            desc = METRIC_DESCRIPTIONS.get(name, f"{name}: see pyiqa documentation.")
            f.write(f"{desc}\n")
        f.write("=" * 120 + "\n\n")

        f.write("Global Summary\n")
        f.write(format_summary("all", global_stats["all"]) + "\n")
        f.write(format_summary("cam0", global_stats["cam0"]) + "\n")
        f.write(format_summary("cam5", global_stats["cam5"]) + "\n")
        f.write("\n")

        f.write("Per Scene Summary\n")
        for scene_id in scene_ids:
            if scene_id not in scene_stats:
                continue
            stats = scene_stats[scene_id]
            if stats["all"].total == 0 and stats["all"].missing == 0:
                continue
            f.write(f"Scene {scene_id}\n")
            f.write(format_summary("all", stats["all"]) + "\n")
            f.write(format_summary("cam0", stats["cam0"]) + "\n")
            f.write(format_summary("cam5", stats["cam5"]) + "\n")
            f.write("\n")


def parse_metric_list(metric_str: str) -> List[str]:
    names = [m.strip().lower() for m in metric_str.split(",") if m.strip()]
    if not names:
        raise ValueError("At least one metric must be specified.")
    valid = set(pyiqa.list_models())
    for name in names:
        if name not in valid:
            raise ValueError(f"Unknown pyiqa metric '{name}'. Run `pyiqa.list_models()` to see available metrics.")
    return names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate NuScenes cam0/cam5 no-reference IQA metrics on rendered images.")
    parser.add_argument("--scene-list", default=DEFAULT_SCENE_LIST, help="Scene id list text file.")
    parser.add_argument("--render-root", default=DEFAULT_RENDER_ROOT, help="Root directory that contains 3cams{scene}fd2/bd2 folders.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Path to the report txt file.")
    parser.add_argument("--cam0-suffix", default=DEFAULT_CAM0_SUFFIX, help="Run suffix used by cam0 scenes.")
    parser.add_argument("--cam5-suffix", default=DEFAULT_CAM5_SUFFIX, help="Run suffix used by cam5 scenes.")
    parser.add_argument(
        "--metrics",
        default=DEFAULT_METRICS,
        help=f"Comma-separated list of pyiqa no-reference metric names. Default: {DEFAULT_METRICS}",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Number of samples processed per GPU batch.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device used for IQA metric computation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_list_path = Path(args.scene_list)
    render_root = Path(args.render_root)
    output_path = Path(args.output)
    device = torch.device(args.device)
    metric_names = parse_metric_list(args.metrics)

    print(f"Metrics to evaluate: {', '.join(metric_names)}")

    scene_ids = load_scene_list(str(scene_list_path))
    if not scene_ids:
        raise FileNotFoundError(f"No scene ids found in {scene_list_path}")

    tasks = build_tasks(
        render_root=render_root,
        scene_ids=scene_ids,
        cam0_suffix=args.cam0_suffix,
        cam5_suffix=args.cam5_suffix,
    )

    if not tasks:
        raise RuntimeError("No valid cam0/cam5 rendered images were found under render-root.")

    global_stats, scene_stats = evaluate_tasks(tasks, metric_names, device, args.batch_size)
    write_report(
        output_path=output_path,
        scene_ids=scene_ids,
        tasks=tasks,
        global_stats=global_stats,
        scene_stats=scene_stats,
        render_root=render_root,
        scene_list_path=scene_list_path,
        cam0_suffix=args.cam0_suffix,
        cam5_suffix=args.cam5_suffix,
        batch_size=args.batch_size,
        metric_names=metric_names,
    )

    print(f"Report saved to {output_path}")
    print(format_summary("all", global_stats["all"]))
    print(format_summary("cam0", global_stats["cam0"]))
    print(format_summary("cam5", global_stats["cam5"]))


if __name__ == "__main__":
    main()
