"""Package project sources, trained models, real inputs and showcase outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
from geo_io import _atomic_write

ROOT = Path(__file__).resolve().parent
ROOT_FILES = ("L1BSR-GUI.py", "L1-BSR-GUI-ONNX.py", "geo_io.py", "gui_support.py", "results_export.py",
              "runtime_support.py", "onnx_engine.py", "data.py", "train.py", "predict.py", "prepare_dataset.py",
              "convert_to_onnx.py", "download_sentinel_demo.py", "generate_demo_images.py", "plot_training.py",
              "prepare_showcase.py", "package_submission.py", "setup_cpu.bat", "run_app.bat", "rcan_model.onnx",
              "download_real_model.py", "evaluate_real_accuracy.py", "REAL_ACCURACY.md", "real_accuracy_report.json",
              "download_hf_model.py", "download_accuracy_samples.py", "evaluate_hf_accuracy.py", "HF_MODEL.md",
              "hf_accuracy_report.json", "hf_accuracy_evaluation.log", "hf_cpu_audit.log", "hf_gpu_audit.log",
              "real_detector_audit.json",
              "real_gpu_audit.log", "real_cpu_audit.log", "real_detector_tests.log",
              "LICENSE", "README.md", "SHOWCASE_GUIDE.md", "PROJECT_REPORT.md", "AUDIT_REPORT.md", "audit_results.json",
              "cpu_audit_results.json", "requirements.txt", "requirements-onnx.txt", "requirements-dev.txt",
              "requirements-tested.txt", "requirements-cpu-tested.txt", ".gitignore")
FOLDERS = ("models", "tests", "checkpoints/demo", "demo_images", "docs", "showcase")


def package(path):
    files = [ROOT/name for name in ROOT_FILES]
    for folder in FOLDERS:
        directory = ROOT/folder
        if not directory.is_dir():
            raise FileNotFoundError(f"Required submission folder missing: {folder}")
        files.extend(file for file in directory.rglob("*") if file.is_file() and
                     "__pycache__" not in file.parts and file.suffix not in (".pyc", ".pyo"))
    for file in files:
        if not file.is_file():
            raise FileNotFoundError(f"Required submission file missing: {file.name}")
    manifest = {file.relative_to(ROOT).as_posix(): hashlib.sha256(file.read_bytes()).hexdigest() for file in files}
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    def write(temp):
        with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for file in sorted(files):
                archive.write(file, "L1BSR_Project/"+file.relative_to(ROOT).as_posix())
            archive.writestr("L1BSR_Project/SUBMISSION_MANIFEST.json", json.dumps(manifest, indent=2))
        with zipfile.ZipFile(temp) as archive:
            if archive.testzip() is not None:
                raise RuntimeError("Submission archive integrity check failed")
    _atomic_write(path, write)
    print(f"Verified {len(files)} files: {path} ({path.stat().st_size/1024**2:.1f} MB)")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT/"submission"/"L1BSR_Submission.zip"))
    package(parser.parse_args().output)
