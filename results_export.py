"""Export a complete GUI result as one atomic ZIP download."""
import json
import tempfile
import zipfile
from pathlib import Path
import numpy as np
from geo_io import (resolve_output_path, save_image, save_text, output_valid_mask,
                    write_geotiff, write_single_band_geotiff, save_jpeg_with_jgw,
                    _atomic_write)


def score_rgb(confidence):
    score = np.nan_to_num(confidence, nan=0.)
    rgb = np.stack([255*(1-score), 255*score, np.zeros_like(score)], -1).astype(np.uint8)
    rgb[~np.isfinite(confidence)] = 0
    return rgb


def export_results_zip(path, original, enhanced_rgb, enhanced_bands, ref, segmentation,
                       report_text, report_data):
    path = resolve_output_path(path, {".zip"}, ".zip")
    if not path.parent.is_dir():
        raise ValueError("Choose an existing download folder")
    valid = output_valid_mask(ref, enhanced_bands.shape)
    with tempfile.TemporaryDirectory(prefix=".l1bsr-download-", dir=path.parent) as name:
        stage = Path(name).resolve()
        if stage.parent != path.parent.resolve():
            raise ValueError("Invalid download workspace")
        save_image(stage/"original.png", original, ref.valid_mask)
        save_image(stage/"enhanced.png", enhanced_rgb, valid)
        save_image(stage/"enhanced.jpg", enhanced_rgb)
        save_image(stage/"land_cover.png", segmentation["color_map"], valid)
        overlay = (enhanced_rgb.astype(np.float32)*.45 +
                   segmentation["color_map"].astype(np.float32)*.55).astype(np.uint8)
        save_image(stage/"overlay.png", overlay, valid)
        save_image(stage/"scores.png", score_rgb(segmentation["confidence_map"]), valid)
        np.save(stage/"class_ids.npy", segmentation["class_map"], allow_pickle=False)
        np.save(stage/"scores.npy", segmentation["confidence_map"], allow_pickle=False)
        np.save(stage/"enhanced_bands.npy", enhanced_bands, allow_pickle=False)
        if ref.crs is not None:
            write_geotiff(stage/"enhanced.tif", enhanced_bands, ref)
            write_single_band_geotiff(stage/"land_cover.tif", segmentation["class_map"], ref)
            save_jpeg_with_jgw(stage/"enhanced.jpg", enhanced_rgb, ref)
        report = dict(report_data)
        report["files"] = sorted([p.name for p in stage.iterdir()] + ["report.md", "report.json"])
        save_text(stage/"report.md", report_text)
        save_text(stage/"report.json", json.dumps(report, indent=2, allow_nan=False))
        def write_zip(temp):
            with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for file in sorted(stage.iterdir()):
                    archive.write(file, file.name)
        _atomic_write(path, write_zip)
    return path
