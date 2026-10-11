# Software audit and regression results

Audit date: 11 October 2026. Scope: both desktop applications, image/raster I/O, models and inference, metrics, preparation/training, command-line prediction, ONNX conversion, CPU/GPU selection, and output publication.

## Real-model update

The default detector now uses real-pretrained OpenEarthMap SparseMask with measured-NIR checks and original-image inference, independently of RCAN. Seven additional tests cover strict weights, author BGR normalization, class mapping, real annotations, spectral gating, tiled coverage, developed-space GeoTIFFs and detector-state changes. The complete suite passed 78/78 checks in the CUDA-capable installation and 76/76 applicable checks in the fresh CPU-only build, with two expected CUDA skips. Full results are in real_detector_audit.json. The 71-test runs below describe the preceding audit; their count is historical. REAL_ACCURACY.md and real_accuracy_report.json distinguish reference agreement, upstream example sanity checks and unknown independent Sentinel-2 accuracy.

## Earlier recorded verification

**71 tests passed; 0 failures, 0 errors, 0 skipped.** The complete suite ran in approximately two minutes on Windows, including CUDA checks on an NVIDIA GeForce RTX 3070 Ti Laptop GPU. GPU verification ran sequentially, without a concurrent preview inference process. Machine-readable results are in `audit_results.json`.

A second complete run in a fresh **PyTorch 2.11.0+cpu** environment passed **69 applicable tests**, with zero failures/errors and two expected CUDA-only skips. `torch.version.cuda` is `None`; this build has no CUDA support. All five real Sentinel-2 scenes completed enhancement, detection and geographic/ZIP exports on CPU. `cpu_audit_results.json` records the run and skips; `requirements-cpu-tested.txt` records this environment, including optional ONNX test dependencies. The Delhi example's loading, enhancement and detection took **3.0 seconds** using four CPU threads on this laptop; other laptops may take longer. Its verified `all_results.zip` and application preview were regenerated using the CPU-only build.

```powershell
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

The GUI tests execute Qt event loops and worker threads offscreen. They exercise the application handlers, preview clicks and actual Qt file-dialog acceptance/cancellation, checking starting directories, shortcuts, remembered folders and output suffixes. Onscreen Windows desktop dialog interaction was not manually verified. Qt dialogs replace the native dialogs to avoid native-dialog re-entry problems.

## Chooser and download accessibility update

The chooser explicitly opens a valid directory, with demo-images, Downloads and Pictures shortcuts and individual image-format filters. It remembers the last image folder. Save dialogs start in Downloads, remember the last save folder, and assign a default suffix before acceptance so overwrite confirmation applies to the actual output file.

All downloads are direct buttons below the preview: enhanced image, class map, GeoTIFF, current view, scores, report and complete ZIP. **Run all (enhance + map)** generates the complete set. Visibility is verified with the application stylesheet at both 1280x800 and 980x640. Complete downloads include arrays and reports, plus geographic files when the source has a CRS. ZIP creation runs in a worker and preserves existing downloads on failure. The seven added tests cover the actual dialogs, control visibility and state, ZIP contents, individual view/score downloads, and failed ZIP-write recovery.

## CPU laptop update

The main app now starts on CPU even when CUDA is available. **Run without GPU (CPU)** is a visible checkbox near the top of the sidebar, with a live device label. Both enhancement and detection use the chosen device. Computers without CUDA keep CPU selected and disable GPU selection. CLI prediction also defaults to CPU; training supports explicit `--device cpu`.

Switching loads both models before changing the active device. The chosen input and model folder are preserved, old enhancement/classification/download state is cleared after success, and old CUDA tensors and cached GPU allocation are released when switching to CPU. Driver or checkpoint errors restore the previous selection and retain its models and downloads. Device selection is disabled and guarded while a worker is active. Five dedicated checks cover CPU operation without any CUDA initialization, CPU defaults, busy guards, rollback, real GPU/CPU inference parity and allocation release, and bounded CPU thread counts.

CPU inference uses no more than four threads and respects a caller's smaller setting. `setup_cpu.bat` installs the official PyTorch 2.11.0+cpu build in a new local environment; `run_app.bat` prefers that environment. The setup script itself is exercised in a clean environment. Copy sources, checkpoints and demo imagery to the other laptop, then create its environment there.

An additional regression fixes a full-range reflectance conversion edge case: multiplying valid uint16 DN by a raster scale and back could exceed 65535 by floating-point roundoff and incorrectly reject the raster. The tolerance accepts only numerical noise, preserves 65535, and still rejects real overflow. The test also decodes an actual tagged GeoTIFF.

| Area | Corrections and verification |
| --- | --- |
| Selecting the next image | Loaded previews remain clickable. Busy controls are restored after the thread has finished. Repeated chooser calls, cancellations, uploads, processing and replacement work within the same app instance. |
| Worker state and memory | A shared worker implementation finalizes thread ownership before running callbacks, clears captured arrays/results/callbacks, handles operation and presentation failures, and guards close while processing. Repeated workflows check Python memory growth below 5 MB; repeated CUDA inference checks retained allocation growth below 1 MB. These are bounded regression checks, not proof of zero memory growth for every workload. |
| Upload validation | Rejects empty, corrupt, truncated, mislabeled and unsupported files; invalid values, dimensions or band layouts; all-transparent/NoData inputs; incompatible satellite grids; and incomplete/duplicate canonical band names. Failed replacement preserves the previous valid scene. File handles are released, verified by renaming loaded files. |
| Photo formats | PNG, JPG/JPEG, BMP, WebP and photo TIF/TIFF support RGB, grayscale, palette, CMYK, alpha and 16-bit cases where the format supports them. EXIF rotation is applied. Alpha is composited for display and represented by a validity mask, never interpreted as NIR. Filename case and Unicode paths are covered. |
| Satellite rasters | Four-band TIFFs and aligned separate bands retain CRS, affine footprint and masks. Canonical band descriptions reorder B02/B03/B04/B08 correctly. Normalized floats and scale/offset-tagged physical reflectance convert consistently to uint16. NoData and invalid float pixels are excluded. TIFF internal masks no longer trigger Pillow's frame parser. |
| Accurate processing | Inputs and checkpoint metadata/weights are validated. RCAN handles single-pixel and odd-size images. Overlapping tiles preserve context; U-Net tile origins align with the pooling grid. PyTorch/ONNX outputs agree within 1 integer DN on the tested sizes, including tiled processing; CPU/CUDA agree within 2 DN. Nonfinite or wrongly shaped model output is rejected. |
| Displays and metrics | Before/after previews use common stretching bounds. PSNR, SSIM, SAM, sharpness and class statistics exclude invalid pixels. Tiny spectra remain scale-invariant in SAM. Class 0 stays a valid label, 255 is NoData. Pixel area uses the affine determinant and projected CRS units. Negative sharpness gains display with their actual sign. |
| State invalidation | Loading a valid new input clears prior enhancement, labels, scores, sharpening and export eligibility. Changing an ONNX input band or model clears stale results and cached sessions appropriately. Failed model replacement preserves the working main-GUI models. |
| Outputs | PNG alpha, GeoTIFF masks/CRS/transforms, class palettes and actual output dimensions are round-tripped. JPEG world files use the affine transform and pixel centers. Individual files are written atomically. JPEG/sidecars and CLI output sets stage and roll back on publication failure. Rerunning CLI prediction removes stale generated geographic/ground-truth outputs while preserving unrelated files. |
| Training and preparation | TIFF/NPZ contracts, label IDs, numeric values, split leakage and consistent batch dimensions are validated before training. Preparation stages a complete dataset before publishing it. A three-scene fixture completes preparation, CPU training, validation and test evaluation. Invalid learning rates and failed preparation leave no partial results. |
| CLI and ONNX export | Prediction works from another working directory and refuses source/output collisions. Invalid input preserves existing outputs. ONNX conversion validates metadata, graph and numerical parity on three shapes before replacing an existing model. Forced failures exercise rollback. |

## Immediate demonstration

`demo_images/` contains **five genuine Sentinel-2 L2A crops**: urban Delhi, Punjab agriculture, Western Ghats vegetation, Pavana water and mixed Pune terrain. Each is a native 256x256 crop with measured B02/B03/B04/B08 bands, real 10 m UTM grid and source acquisition metadata. Aligned individual bands are under `satellite_bands/<area>/`. STAC JSON, source URLs, crop windows and file/source-DN SHA-256 checksums are included. No generated images or ordinary photographs appear in the app's demo list.

```powershell
python L1BSR-GUI.py
```

Select a **Sentinel-2** sample, press **Run all**, inspect the views, and use the visible Downloads panel. Select another real scene immediately using **Open image** or a click on the displayed preview.

For the full CPU-only enhancement, detection and export pipeline, leave **Run without GPU (CPU)** checked in the main app. On another Windows laptop, run `setup_cpu.bat` once, then `run_app.bat`.

For the secondary CPU-only super-resolution app:

```powershell
python L1-BSR-GUI-ONNX.py
```

The included ONNX graph and PyTorch checkpoints make both demonstrations ready to run after dependency installation. All five real scenes complete main-GUI loading, enhancement, detection, score/view generation, image/map/report exports and complete ZIP download. All five also complete ONNX processing and GeoTIFF export. Tests verify 5 m output sampling, identical geographic footprint, acquisition/source tags, source pixels, original native grid, measured NIR and aligned separate-band loading.

`outputs/real_satellite/urban_delhi/all_results.zip` contains a complete result from genuine Delhi imagery. The main and ONNX app previews use that real scene, and `docs/sentinel2_dataset.png` displays all five real source crops. They were visually inspected.

The 26 generated software format fixtures are now isolated in `tests/fixtures/formats/`. They remain useful for codec, malformed-input and edge-case regressions and are never offered as satellite demo data. `generate_demo_images.py` targets that fixture directory and refuses to overwrite `demo_images/`. The real-source downloader is `download_sentinel_demo.py`.

Radiometry is checked against preserved source DN arrays and original STAC metadata. These COGs have `earthsearch:boa_offset_applied=true`; their effective offset is zero, preventing a second subtraction of the JP2 offset retained in the legacy catalog. Calibration and attribution are recorded in the manifest and GeoTIFFs. A source NoData pixel in the vegetation crop is correctly masked, with label 255 and NaN score, while its original on-disk DN values remain preserved. SCL is used for quality screening, never as a five-class accuracy target.

## Supported boundaries

- Animated inputs and multi-page TIFFs are rejected; select/export a single frame first.
- Ordinary photos have a 40-million-pixel decode limit. Inference inputs are capped at 1024 pixels on the longest side, with raster transforms adjusted for resampling.
- Ordinary RGB/gray inputs do not contain measured NIR. The real detector uses RGB only; the synthetic detector remains illustrative on those inputs. Geographic area is unavailable without a suitable projected CRS.
- Demo inputs are real Sentinel-2 data. The default detector is real-pretrained; RCAN and optional U-Net weights are synthetic-trained, and software format fixtures are generated. Regression tests verify source preservation and software behavior; no independent HR reference or land-cover ground truth is included, so real reconstruction/classification accuracy is not established.
- RCAN channel attention depends on tile context; tiled and whole-image inference can differ. The parity thresholds above compare implementations using the same tiling policy, not a guarantee of identical full-scene predictions.

The tests cover the identified defects and principal supported workflows. They cannot guarantee the absence of every possible bug or platform-specific issue.
