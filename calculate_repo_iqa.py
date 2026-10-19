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

# Default paths based on user input
DEFAULT_SCENE_LIST = "nuScenes_Val.txt"
DEFAULT_IMAGE_ROOT = "nuscenes_stitched_results_with_crop_stats0130"
DEFAULT_OUTPUT = "iqa_report_repo_all_0.5.txt"
DEFAULT_BATCH_SIZE = 1
DEFAULT_METRICS = "niqe,brisque,musiq,clipiqa,topiq_nr,arniqa,liqe"

WHITE_PIXEL_THRESHOLD = 0.9
WHITE_AREA_RATIO = 0.5
# 1.1 means calculate all images

# qalign # export HF_ENDPOINT=https://hf-mirror.com
# conda activate iqa
# CUDA_VISIBLE_DEVICES=1 python calculate_repo_iqa.py --metrics niqe,brisque,musiq
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
        self.quality_failed: int = 0

    def update(self, values: Dict[str, float]) -> None:
        for name in self._metric_names:
            self._values[name].append(values[name])
        self.total += 1

    def mark_missing(self) -> None:
        self.missing += 1

    def mark_quality_failed(self) -> None:
        self.quality_failed += 1

    def mean(self) -> Dict[str, float]:
        result: Dict[str, float] = {}
        for name in self._metric_names:
            vals = self._values[name]
            result[name] = float(np.nanmean(vals)) if vals else 0.0
        result["count"] = len(self._values[self._metric_names[0]]) if self._metric_names else 0
        result["missing"] = self.missing
        result["quality_failed"] = self.quality_failed
        return result

    @property
    def metric_names(self) -> List[str]:
        return list(self._metric_names)

def load_scene_list(txt_path: str) -> List[str]:
    if not os.path.exists(txt_path):
        return []
    with open(txt_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def load_rgb_tensor(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    return transforms.ToTensor()(image)

def check_stitching_quality(img_tensor: torch.Tensor) -> bool:
    C, H, W = img_tensor.shape
    quarter_w = W // 4
    if quarter_w == 0:
        return True

    left_region = img_tensor[:, :, :quarter_w]
    right_region = img_tensor[:, :, -quarter_w:]

    left_white_mask = (left_region.mean(dim=0) > WHITE_PIXEL_THRESHOLD).float()
    right_white_mask = (right_region.mean(dim=0) > WHITE_PIXEL_THRESHOLD).float()

    left_ratio = left_white_mask.mean().item()
    right_ratio = right_white_mask.mean().item()

    if left_ratio > WHITE_AREA_RATIO or right_ratio > WHITE_AREA_RATIO:
        return False
    return True

def make_stats_dict(metric_names: Sequence[str]) -> Dict[str, MetricSummary]:
    return {
        "all": MetricSummary(metric_names),
        "front": MetricSummary(metric_names),
        "back": MetricSummary(metric_names),
    }

def format_summary(name: str, summary: MetricSummary) -> str:
    stats = summary.mean()
    parts = [f"{name:<10}", f"N={stats['count']:5d}", f"Missing={stats['missing']:5d}", f"QualityFail={stats['quality_failed']:5d}"]
    for metric_name in summary.metric_names:
        parts.append(f"{metric_name.upper()}={stats[metric_name]:.4f}")
    return " | ".join(parts)

def build_tasks(
    image_root: Path,
    scene_ids: Sequence[str],
) -> List[Dict[str, object]]:
    tasks: List[Dict[str, object]] = []

    for scene_id in scene_ids:
        scene_dir = image_root / scene_id
        if not scene_dir.exists():
            print(f"Warning: Scene directory {scene_dir} not found.")
            continue

        for file_name in sorted(os.listdir(scene_dir)):
            if file_name.endswith("wide.jpg"):
                # Example: 000_front_wide.jpg or 000_back_wide.jpg
                path = scene_dir / file_name
                stem = Path(file_name).stem
                parts = stem.split("_")
                
                # We expect {frame_id}_{view_type}_wide
                if len(parts) >= 2:
                    view_type = parts[-2] # 'front' or 'back'
                    frame_id = "_".join(parts[:-2])
                else:
                    view_type = "unknown"
                    frame_id = stem

                tasks.append({
                    "scene_id": scene_id,
                    "frame_id": frame_id,
                    "view_type": view_type,
                    "path": path,
                })

    return tasks

def evaluate_tasks(
    tasks: Sequence[Dict[str, object]],
    metric_names: Sequence[str],
    device: torch.device,
    batch_size: int,
) -> Tuple[Dict[str, MetricSummary], Dict[str, Dict[str, MetricSummary]]]:
    metric_fns: Dict[str, object] = {}
    for name in metric_names:
        print(f"Loading metric: {name}")
        metric_fns[name] = pyiqa.create_metric(name, device=device)
        if hasattr(metric_fns[name], "eval") and callable(getattr(metric_fns[name], "eval")):
            metric_fns[name].eval()

    global_stats = make_stats_dict(metric_names)
    scene_stats: Dict[str, Dict[str, MetricSummary]] = defaultdict(lambda: make_stats_dict(metric_names))

    # Sort tasks for consistency
    sorted_tasks = sorted(
        tasks,
        key=lambda x: (x["scene_id"], x["frame_id"], x["view_type"])
    )

    for start_idx in tqdm(range(0, len(sorted_tasks), batch_size), desc="Evaluating"):
        batch_tasks = sorted_tasks[start_idx : start_idx + batch_size]

        valid_tasks: List[Dict[str, object]] = []
        pred_list: List[torch.Tensor] = []

        for task in batch_tasks:
            try:
                pred_tensor = load_rgb_tensor(task["path"])
            except Exception as e:
                print(f"Error loading {task['path']}: {e}")
                view_key = task["view_type"] if task["view_type"] in ["front", "back"] else "all"
                global_stats[view_key].mark_missing()
                global_stats["all"].mark_missing()
                scene_stats[task["scene_id"]][view_key].mark_missing()
                scene_stats[task["scene_id"]]["all"].mark_missing()
                continue

            if not check_stitching_quality(pred_tensor):
                view_key = task["view_type"] if task["view_type"] in ["front", "back"] else "all"
                global_stats[view_key].mark_quality_failed()
                global_stats["all"].mark_quality_failed()
                scene_stats[task["scene_id"]][view_key].mark_quality_failed()
                scene_stats[task["scene_id"]]["all"].mark_quality_failed()
                continue

            valid_tasks.append(task)
            pred_list.append(pred_tensor)

        if not valid_tasks:
            continue

        # Group by size for batching (though most images might be same size)
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
                    scores: List[float] = []
                    for i in range(pred_batch.shape[0]):
                        single_img = pred_batch[i:i+1]
                        try:
                            score = fn(single_img)
                            scores.append(torch.atleast_1d(score).detach().cpu().reshape(-1).item())
                        except Exception as e:
                            print(f"Warning: {name} failed for image: {e}")
                            scores.append(float('nan'))
                    metric_batches[name] = torch.tensor(scores, dtype=torch.float32)

            for idx, task in enumerate(group_tasks):
                scene_id = task["scene_id"]
                view_type = task["view_type"]
                view_key = view_type if view_type in ["front", "back"] else "all"

                values = {name: metric_batches[name][idx].item() for name in metric_names}

                global_stats[view_key].update(values)
                global_stats["all"].update(values)
                scene_stats[scene_id][view_key].update(values)
                scene_stats[scene_id]["all"].update(values)

    return global_stats, scene_stats

def write_report(
    output_path: Path,
    scene_ids: Sequence[str],
    tasks: Sequence[Dict[str, object]],
    global_stats: Dict[str, MetricSummary],
    scene_stats: Dict[str, Dict[str, MetricSummary]],
    image_root: Path,
    scene_list_path: Path,
    metric_names: Sequence[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Repository IQA Metric Report (Stitched Wide Images)\n")
        f.write(f"Scene list: {scene_list_path}\n")
        f.write(f"Image root: {image_root}\n")
        f.write(f"Metrics: {', '.join(metric_names)}\n")
        f.write(f"Total tasks collected: {len(tasks)}\n")
        f.write("=" * 120 + "\n\n")

        f.write("Global Summary\n")
        all_stats = global_stats["all"].mean()
        total_collected = len(tasks)
        total_evaluated = all_stats["count"]
        total_missing = all_stats["missing"]
        total_quality_failed = all_stats["quality_failed"]
        f.write(f"Total images found: {total_collected}\n")
        f.write(f"  - Load failed:       {total_missing}\n")
        f.write(f"  - Quality failed:    {total_quality_failed} (White borders detected)\n")
        f.write(f"  - Evaluated:         {total_evaluated}\n\n")
        f.write(format_summary("Overall", global_stats["all"]) + "\n")
        f.write(format_summary("Front View", global_stats["front"]) + "\n")
        f.write(format_summary("Back View", global_stats["back"]) + "\n")
        f.write("\n")

        f.write("Per Scene Summary\n")
        for scene_id in scene_ids:
            if scene_id not in scene_stats:
                continue
            stats = scene_stats[scene_id]
            if stats["all"].total == 0 and stats["all"].missing == 0:
                continue
            f.write(f"Scene {scene_id}\n")
            f.write(format_summary("  Total", stats["all"]) + "\n")
            f.write(format_summary("  Front", stats["front"]) + "\n")
            f.write(format_summary("  Back", stats["back"]) + "\n")
            f.write("-" * 80 + "\n")

        f.write("\nMetric Descriptions:\n")
        for name in metric_names:
            desc = METRIC_DESCRIPTIONS.get(name, f"{name}: see pyiqa documentation.")
            f.write(f"- {desc}\n")

def parse_metric_list(metric_str: str) -> List[str]:
    names = [m.strip().lower() for m in metric_str.split(",") if m.strip()]
    if not names:
        raise ValueError("At least one metric must be specified.")
    valid = set(pyiqa.list_models())
    for name in names:
        if name not in valid:
            raise ValueError(f"Unknown pyiqa metric '{name}'.")
    return names

def main():
    parser = argparse.ArgumentParser(description="Calculate IQA metrics for stitched wide images.")
    parser.add_argument("--scene-list", default=DEFAULT_SCENE_LIST, help="Scene id list text file.")
    parser.add_argument("--image-root", default=DEFAULT_IMAGE_ROOT, help="Root directory for stitched images.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Path to the report txt file.")
    parser.add_argument("--metrics", default=DEFAULT_METRICS, help="Comma-separated list of pyiqa metrics.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    scene_list_path = Path(args.scene_list)
    image_root = Path(args.image_root)
    output_path = Path(args.output)
    device = torch.device(args.device)
    metric_names = parse_metric_list(args.metrics)

    print(f"Loading scene list from {scene_list_path}...")
    scene_ids = load_scene_list(str(scene_list_path))
    if not scene_ids:
        print(f"Error: No scene IDs found in {scene_list_path}")
        return

    print("Building tasks...")
    tasks = build_tasks(image_root, scene_ids)
    if not tasks:
        print(f"Error: No images ending in 'wide.jpg' found in {image_root}")
        return
    print(f"Found {len(tasks)} images to evaluate.")

    global_stats, scene_stats = evaluate_tasks(tasks, metric_names, device, args.batch_size)

    print("\n=== Evaluation Results ===")
    print(format_summary("Overall", global_stats["all"]))
    print(format_summary("Front View", global_stats["front"]))
    print(format_summary("Back View", global_stats["back"]))

    print(f"Writing report to {output_path}...")
    write_report(
        output_path=output_path,
        scene_ids=scene_ids,
        tasks=tasks,
        global_stats=global_stats,
        scene_stats=scene_stats,
        image_root=image_root,
        scene_list_path=scene_list_path,
        metric_names=metric_names,
    )
    print("Done.")

if __name__ == "__main__":
    main()
