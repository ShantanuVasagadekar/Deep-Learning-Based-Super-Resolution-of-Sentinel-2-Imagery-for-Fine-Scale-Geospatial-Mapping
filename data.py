"""Reproducible synthetic scenes and explicit train/val/test NPZ datasets."""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import zipfile
from torch.utils.data import Dataset
from models.common import extract_features, validate_image, NORMALIZATION


def synthetic_scene(seed, size=64):
    """Procedural illustration, never an observed Sentinel-2 scene."""
    if type(size) is not int or size < 16 or size % 2:
        raise ValueError("Synthetic scene size must be even and at least 16")
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:size, :size]
    mask = np.full((size, size), 4, dtype=np.uint8)
    # Fields/forest, a river, building blocks and intersecting roads.
    for _ in range(8):
        cy, cx = rng.integers(0, size, 2)
        ry, rx = rng.integers(max(3, size//8), max(4, size//3), 2)
        region = ((y-cy)/ry)**2 + ((x-cx)/rx)**2 < 1
        mask[region] = 2
    center = rng.uniform(.15, .85)*size + size*.1*np.sin(y/rng.uniform(5, 15))
    mask[np.abs(x-center) < rng.uniform(2, 5)] = 3
    for _ in range(16):
        by, bx = rng.integers(0, size-8, 2)
        bh, bw = rng.integers(4, 12, 2)
        mask[by:by+bh, bx:bx+bw] = 0
    road_x, road_y = rng.integers(size//4, 3*size//4, 2)
    mask[(np.abs(x-road_x) <= 1) | (np.abs(y-road_y) <= 1)] = 1
    spectra = np.array([[2400, 2600, 3100, 2900], [1400, 1450, 1500, 1650],
                        [380, 720, 420, 4200], [1350, 850, 420, 160],
                        [1250, 1650, 2100, 2650]], dtype=np.float32)
    spectra *= rng.uniform(.8, 1.2, (5, 1))
    hr = spectra[mask] + rng.normal(0, 30, (size, size, 4))
    hr = np.clip(hr, 0, 10000).round().astype(np.uint16)
    lr = degrade(hr)
    return lr, hr, mask


def degrade(hr):
    hr = validate_image(hr)
    if min(hr.shape[:2]) < 2 or any(v % 2 for v in hr.shape[:2]):
        raise ValueError("HR degradation requires even dimensions of at least 2")
    tensor = torch.from_numpy(hr.astype(np.float32)).permute(2, 0, 1)[None]
    out = F.interpolate(tensor, size=(hr.shape[0]//2, hr.shape[1]//2),
                        mode="bicubic", align_corners=False, antialias=True)[0]
    return np.clip(np.moveaxis(out.numpy(), 0, -1), 0, 65535).round().astype(np.uint16)


class SatelliteDataset(Dataset):
    """NPZ: hr [2H,2W,4], lr [H,W,4], mask [2H,2W]; uint16 bands, uint8 IDs."""
    def __init__(self, root=None, split="train", demo=False, samples=128, size=64):
        if split not in ("train", "val", "test"):
            raise ValueError("split must be train, val or test")
        if demo and (type(samples) is not int or not 1 <= samples <= 90000):
            raise ValueError("Demo samples must be 1..90000 to keep split seed ranges disjoint")
        self.demo, self.size = demo, size
        self.seed = {"train": 1000, "val": 100000, "test": 200000}[split]
        self.files = sorted((Path(root)/split).glob("*.npz")) if root else []
        self.samples = samples if demo else len(self.files)
        if not self.samples:
            raise ValueError(f"No .npz scenes in {root}/{split}.")
        if demo and (type(size) is not int or size < 16 or size % 2 or size > 512):
            raise ValueError("Synthetic scene size must be even and at least 16.")

    def __len__(self):
        return self.samples

    def arrays(self, index):
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError("Scene index out of range")
        if self.demo:
            return synthetic_scene(self.seed + index, self.size)
        with zipfile.ZipFile(self.files[index]) as archive:
            if sum(info.file_size for info in archive.infolist()) > 64*1024*1024:
                raise ValueError("NPZ scene exceeds the 64 MB decompressed training-patch limit")
        with np.load(self.files[index], allow_pickle=False) as scene:
            if not {"hr", "mask"} <= set(scene.files):
                raise ValueError("Each NPZ scene requires hr and mask arrays")
            hr = validate_image(scene["hr"])
            lr = validate_image(scene["lr"]) if "lr" in scene else degrade(hr)
            mask = scene["mask"]
        if hr.dtype != np.uint16 or lr.dtype != np.uint16:
            raise ValueError("Real bands must be uint16 reflectance scaled by 10000.")
        if hr.shape[:2] != (lr.shape[0]*2, lr.shape[1]*2) or mask.shape != hr.shape[:2]:
            raise ValueError("HR and mask must be exactly 2x LR dimensions.")
        if min(lr.shape[:2]) < 8 or mask.dtype.kind not in "ui" or not np.isin(mask, [0, 1, 2, 3, 4, 255]).all():
            raise ValueError("LR must be at least 8x8 and mask IDs must be 0..4 or 255 (ignore).")
        if not (mask != 255).any():
            raise ValueError("Scene has no labeled pixels.")
        return lr, hr, mask

    def __getitem__(self, index):
        lr, hr, mask = self.arrays(index)
        features, _ = extract_features(hr)
        return (torch.from_numpy(np.moveaxis(lr.astype(np.float32)/NORMALIZATION, -1, 0).copy()),
                torch.from_numpy(np.moveaxis(hr.astype(np.float32)/NORMALIZATION, -1, 0).copy()),
                torch.from_numpy(np.moveaxis(features, -1, 0).copy()),
                torch.from_numpy(mask.astype(np.int64)))
