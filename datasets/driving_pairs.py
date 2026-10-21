"""Two-image driving pairs for the original REwarp training loop.

Each sample is camera 5 plus either camera 4 or camera 3, resized to the
512 square the H/T cells were trained on. One epoch shuffles every pair
unless ``samples_per_epoch`` is set. ``lyft`` mixes lyft1920 and lyft1224.
"""
import os
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from datasets import register

SIDES = (4, 3)
IMAGE_EXTS = (".jpg", ".png", ".jpeg")

DATASETS = {
    "nuscenes": {
        "train_images": "nuscenes/processed_10Hz_v2",
        "train_list": "nuscenes/processed_10Hz_v2/nuScenes_Train.txt",
        "val_images": "nuscenes/processed_10Hz_v2",
        "val_list": "nuscenes/processed_10Hz_v2/nuScenes_Val.txt",
    },
    "lyft1920": {
        "train_images": "lyft/lyft_train1920_3cams",
        "train_list": "lyft/lyft_train1920_3cams/train1920.txt",
        "val_images": "lyft/lyft_val1920_3cams",
        "val_list": "lyft/lyft_val1920_3cams/lyft_val1920.txt",
    },
    "lyft1224": {
        "train_images": "lyft/lyft_train1224_3cams",
        "train_list": "lyft/lyft_train1224_3cams/train1224.txt",
        "val_images": "lyft/lyft_val1224_3cams",
        "val_list": "lyft/lyft_val1224_3cams/lyft_val1224.txt",
    },
    # WideDriveTrain sits next to WideDriveVal and uses the same scene layout.
    "widedrive3": {
        "train_images": "WideDrive3_processed/WideDriveTrain",
        "train_list": "WideDrive3_processed/WideDriveTrain/train.txt",
        "val_images": "WideDrive3_processed/WideDriveVal",
        "val_list": "WideDrive3_processed/WideDriveVal/val.txt",
    },
}
GROUPS = {
    "lyft": ("lyft1920", "lyft1224"),
}


def default_data_root():
    env = os.environ.get("REWARP_DATA_ROOT")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "RopStitch_infer" / "datasets"


def _read_list(path):
    with open(path) as handle:
        return [line.strip() for line in handle if line.strip()]


def _find(image_dir, stem, cam):
    for ext in IMAGE_EXTS:
        path = image_dir / f"{stem}_{cam}{ext}"
        if path.is_file():
            return path
    return None


@register("driving-camera-pairs")
class DrivingCameraPairs(Dataset):
    def __init__(
        self,
        dataset,
        split="train",
        data_root=None,
        image_size=512,
        samples_per_epoch=0,
        seed=2022,
    ):
        names = GROUPS.get(dataset, (dataset,))
        unknown = [name for name in names if name not in DATASETS]
        if unknown:
            known = ", ".join(sorted(DATASETS) + sorted(GROUPS))
            raise ValueError(f"no training split for {dataset}; available: {known}")
        root = Path(data_root) if data_root else default_data_root()
        self.image_size = int(image_size)
        self.samples_per_epoch = int(samples_per_epoch)
        self.seed = int(seed)
        self.split = split
        self.pairs = []
        for name in names:
            spec = DATASETS[name]
            list_path = root / spec[f"{split}_list"]
            image_root = root / spec[f"{split}_images"]
            if not list_path.is_file():
                raise FileNotFoundError(f"missing {split} list {list_path}")
            size = input_hw(name)
            self.pairs.extend((ref, tgt, size) for ref, tgt in _collect(image_root, _read_list(list_path)))
        if not self.pairs:
            raise RuntimeError(f"no camera 5/4 or 5/3 pairs for {dataset}")
        self.set_epoch(0)

    def set_epoch(self, epoch):
        rng = np.random.RandomState(self.seed + int(epoch))
        order = rng.permutation(len(self.pairs))
        count = len(self.pairs)
        if 0 < self.samples_per_epoch < count:
            count = self.samples_per_epoch
        self.index = order[:count]

    def __len__(self):
        return len(self.index)

    def current_sizes(self):
        return [self.pairs[int(i)][2] for i in self.index]

    def __getitem__(self, idx):
        ref_path, tgt_path, size = self.pairs[int(self.index[idx])]
        return _load(ref_path, size), _load(tgt_path, size)


def _collect(image_root, scenes):
    pairs = []
    for scene in scenes:
        image_dir = Path(image_root) / scene / "images"
        if not image_dir.is_dir():
            continue
        names = set(os.listdir(image_dir))
        stems = []
        for name in names:
            for ext in IMAGE_EXTS:
                suffix = f"_5{ext}"
                if name.endswith(suffix):
                    stems.append(name[: -len(suffix)])
                    break
        for stem in stems:
            ref = _find(image_dir, stem, 5)
            if ref is None:
                continue
            for side in SIDES:
                path = _find(image_dir, stem, side)
                if path is not None:
                    pairs.append((str(ref), str(path)))
    return pairs


def input_hw(name):
    """Width, height used before the network, matching the compared model."""
    if name == "lyft1224":
        return (518, 434)
    return (518, 294)


def _load(path, size):
    image = Image.open(path).convert("RGB")
    if isinstance(size, int):
        image = image.resize((size, size), Image.BILINEAR)
    else:
        if image.size != tuple(size):
            image = image.resize(tuple(size), Image.BILINEAR)
        image = image.resize((512, 512), Image.BILINEAR)
    array = np.asarray(image).astype(np.float32) / 255.0
    return array
