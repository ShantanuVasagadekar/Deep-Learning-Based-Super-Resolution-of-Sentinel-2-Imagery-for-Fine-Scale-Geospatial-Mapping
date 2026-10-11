"""Train RCAN and U-Net, select on validation, and evaluate untouched test scenes."""
import argparse
import json
import random
import importlib.metadata
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from data import SatelliteDataset
from models.rcan import RCAN
from models.unet_segmentation import SatelliteSegmentationNet
from models.common import BAND_ORDER, NORMALIZATION
from models.evaluation_metrics import reconstruction_metrics, segmentation_metrics
from geo_io import _atomic_write


def save_checkpoint(path, model, task, config, source, epoch):
    checkpoint = dict(format_version=1, task=task, model_config=config,
        state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
        band_order=BAND_ORDER, normalization=NORMALIZATION,
        data_source=source, epoch=epoch)
    _atomic_write(Path(path), lambda temporary: torch.save(checkpoint, temporary))


@torch.inference_mode()
def validate(model, loader, task, device):
    model.eval()
    loss, count = 0.0, 0
    for lr, hr, features, mask in loader:
        target = hr.to(device) if task == "sr" else mask.to(device)
        output = model((lr if task == "sr" else features).to(device))
        value = torch.nn.functional.l1_loss(output, target) if task == "sr" else torch.nn.functional.cross_entropy(output, target, ignore_index=255)
        if not torch.isfinite(value):
            raise RuntimeError("Non-finite validation loss; check input data")
        weight = target.numel() if task == "sr" else int((target != 255).sum())
        loss += value.item()*weight
        count += weight
    return loss/count


def train_task(task, train_loader, val_loader, args, device, source):
    config = dict(n_colors=4, scale=2, n_feats=32, n_resgroups=3, n_resblocks=4, residual=True) if task == "sr" else dict(in_channels=8, num_classes=5)
    model = (RCAN(**config) if task == "sr" else SatelliteSegmentationNet(**config)).to(device)
    if task == "sr":
        # Start at the bicubic baseline and learn the residual through gradient descent.
        torch.nn.init.zeros_(model.tail[-1].weight)
        torch.nn.init.zeros_(model.tail[-1].bias)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history, best = [], float("inf")
    filename = Path(args.output)/("rcan.pt" if task == "sr" else "unet.pt")
    for epoch in range(1, args.epochs+1):
        model.train()
        total, count = 0.0, 0
        for lr, hr, features, mask in train_loader:
            optimizer.zero_grad(set_to_none=True)
            output = model((lr if task == "sr" else features).to(device))
            target = hr.to(device) if task == "sr" else mask.to(device)
            loss = torch.nn.functional.l1_loss(output, target) if task == "sr" else torch.nn.functional.cross_entropy(output, target, ignore_index=255)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss; check dataset scaling and labels.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            weight = target.numel() if task == "sr" else int((target != 255).sum())
            total += loss.item()*weight
            count += weight
        val_loss = validate(model, val_loader, task, device)
        history.append(dict(epoch=epoch, train_loss=total/count, val_loss=val_loss))
        if val_loss < best:
            best = val_loss
            save_checkpoint(filename, model, task, config, source, epoch)
        print(f"{task} epoch {epoch:03d}/{args.epochs}: train={total/count:.5f} val={val_loss:.5f}", flush=True)
    return history


def preflight_datasets(datasets):
    """Fail before training on bad patches, mixed batch sizes or exact split leakage."""
    seen = {}
    for split, dataset in datasets.items():
        shape = None
        for index in range(len(dataset)):
            lr, hr, mask = dataset.arrays(index)
            if shape is None:
                shape = hr.shape
            if hr.shape != shape:
                raise ValueError(f"{split} patches must have the same dimensions for batching")
            if not dataset.demo:
                fingerprint = hashlib.sha256(hr.tobytes()).digest()
                if fingerprint in seen and seen[fingerprint] != split:
                    raise ValueError(f"Identical HR imagery appears in {seen[fingerprint]} and {split}; split leakage")
                seen[fingerprint] = split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo", action="store_true")
    source.add_argument("--data", help="Directory containing train/, val/, test/ NPZ scenes")
    parser.add_argument("--output", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.epochs < 1 or args.samples < 1 or args.batch_size < 1 or not np.isfinite(args.lr) or args.lr <= 0:
        parser.error("epochs, samples, batch-size and lr must be positive")
    args.output = args.output or ("checkpoints/demo" if args.demo else "checkpoints/real")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    Path(args.output).mkdir(parents=True, exist_ok=True)
    datasets = {split: SatelliteDataset(args.data, split, args.demo,
                 args.samples if split == "train" else max(16, args.samples//4), args.size)
                for split in ("train", "val", "test")}
    preflight_datasets(datasets)
    loaders = {split: DataLoader(ds, batch_size=args.batch_size, shuffle=split == "train", num_workers=0)
               for split, ds in datasets.items()}
    source_name = "synthetic procedural demo (not real satellite validation)" if args.demo else "user supplied NPZ scenes (verify source independently)"
    history = {task: train_task(task, loaders["train"], loaders["val"], args, device, source_name)
               for task in ("sr", "segmentation")}
    # Select only by validation loss. Test scenes are evaluated once at the end.
    from models.rcan import L1BSRSR
    from models.unet_segmentation import LandCoverSegmenter
    sr = L1BSRSR(Path(args.output)/"rcan.pt", str(device))
    seg = LandCoverSegmenter(str(device), Path(args.output)/"unet.pt")
    recon, baseline, confusion = [], [], np.zeros((5, 5), dtype=np.int64)
    direct_confusion = np.zeros((5, 5), dtype=np.int64)
    for index in range(len(datasets["test"])):
        lr, hr, mask = datasets["test"].arrays(index)
        prediction = sr.super_resolve(lr)
        recon.append(reconstruction_metrics(prediction, hr, max_val=NORMALIZATION))
        baseline.append(reconstruction_metrics(sr.bicubic_upscale(lr), hr, max_val=NORMALIZATION))
        confusion += np.array(segmentation_metrics(seg.segment(prediction, apply_morphology=False, return_probabilities=False)["class_map"], mask)["confusion_matrix"])
        direct_confusion += np.array(segmentation_metrics(seg.segment(hr, apply_morphology=False, return_probabilities=False)["class_map"], mask)["confusion_matrix"])
    report = dict(data_source=source_name, settings=vars(args), split_sizes={k:len(v) for k,v in datasets.items()},
        history=history, test=dict(sr={k:float(np.mean([v[k] for v in recon])) for k in recon[0]},
        bicubic={k:float(np.mean([v[k] for v in baseline])) for k in baseline[0]},
        segmentation_pipeline=segmentation_metrics(confusion=confusion),
        segmentation_on_hr=segmentation_metrics(confusion=direct_confusion)))
    report["environment"] = {key: importlib.metadata.version(key) for key in
                             ("torch", "numpy", "rasterio", "PyQt5", "Pillow", "scipy")}
    report["evaluation_protocol"] = "Select by validation loss; evaluate held-out test once; no morphology."
    (Path(args.output)/"training_report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report["test"], indent=2), flush=True)


if __name__ == "__main__":
    main()
