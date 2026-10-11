# Tomorrow's project showcase

Project: Deep Learning-Based Super-Resolution Mapping from Medium-Resolution Satellite Imagery.

## What is working

- Genuine four-band Sentinel-2 imagery is loaded with measured blue, green, red and near-infrared data and real georeferencing.
- A trained RCAN produces a 2x image: the included 256x256 samples become 512x512. Its predictions differ from bicubic interpolation.
- The default real-pretrained OpenEarthMap network assigns pixels to buildings, roads, vegetation, water, bare land and developed space. Detection uses the original image independently of enhancement. The Map and Map on photo views show those predictions.
- Vegetation includes trees, grass and crops. There are no individual tree counts or building bounding boxes.
- CPU mode supports the full pipeline on a laptop, without CUDA. Downloads include images, predicted labels, scores, reports and georeferenced rasters.
- Another image can be selected and processed without restarting. Five real sample scenes are included, with offline results in showcase/.

## What the results establish

This is a complete runnable learning prototype with training code, trained checkpoints, held-out synthetic evaluation, real-data preparation, real satellite inference, a desktop GUI and export workflows.

RCAN and the optional U-Net were trained on synthetic scenes. The default detector is pretrained on real manually annotated OpenEarthMap imagery at 0.25-0.5 m resolution. It is not validated for precise roads/buildings in 10 m Sentinel-2 data. The real Sentinel-2 samples have no independently labeled masks, so their classification accuracy remains unknown. See REAL_ACCURACY.md and real_accuracy_report.json.

The default now combines **Hugging Face SegFormer and SparseMask**. It achieves **79.80% pixel accuracy on eight fixed LoveDA audit images**, versus 77.17% for SparseMask alone. The 85% target has not been met. Two OEM examples improve from 80.91% to 84.11%, but may overlap training and are only sanity checks. Roads, bare ground and small objects still fail. See HF_MODEL.md for the exact class mapping, excluded background pixels, per-class results and limitations. Do not present these figures as accuracy on arbitrary uploads.

The RCAN predicts an enlarged image and may make edges look sharper. Its output is sampled at 5 m from a 10 m input, but the project has not established true 5 m resolving power or recovered building details. GUI PSNR/SSIM/SAM compare with bicubic, not a real high-resolution reference. Extra sharpening is a display filter.

On the held-out synthetic experiment, RCAN PSNR was 35.74 dB versus 30.08 dB for bicubic. Full pipeline pixel accuracy was 98.13% and mean IoU 94.38%. Say **synthetic test data** every time these numbers are presented.

## Preparation tonight

1. Extract submission/L1BSR_Submission.zip into a normal writable folder on the demonstration laptop. Install 64-bit Python with Python on PATH if needed.
2. Run setup_cpu.bat while internet is available. It creates the CPU environment locally. Do not copy a virtual environment between computers.
3. Run run_app.bat. Leave Run without GPU (CPU) checked.
4. Keep Combined real models selected, or choose Hugging Face SegFormer (fast) on a slower laptop. Start with Pavana or the fine-resolution Palu sample and press Run all. Check every view and export All results ZIP to a folder you can locate.
5. Repeat with the Pavana water sample, then select another scene to show that no app restart is required.
6. Keep showcase/overview.png, each scene's results.zip and PROJECT_REPORT.md accessible as offline backups. The real detector adds CPU processing time; allow several seconds for small crops and longer for larger scenes. For a road/building example, open demo_images/high_resolution/palu_8.tif: it is genuine high-resolution Maxar/xBD satellite imagery with original EPSG:4326 georeferencing. It has no measured NIR, so spectral checks are disabled; geographic exports retain its footprint, with physical area unavailable for geographic coordinates.

## Five-minute live demonstration

| Time | Action | Explain |
| --- | --- | --- |
| 0:00-0:40 | Open app, show CPU checkbox and real sample list | Four measured bands; source data and trained models are included offline. |
| 0:40-1:30 | Choose Pavana; Run all | RCAN reconstructs a 2x image; real-pretrained OpenEarthMap detects from the original image. |
| 1:30-2:20 | Enhanced and Compare views | Both views use the same display stretch. Increased sharpness is not verified new ground detail. |
| 2:20-3:10 | Map and Map on photo | Colors correspond to predicted classes. Point out plausible regions and visible mistakes honestly. |
| 3:10-3:40 | Scores and Details | Scores are uncalibrated model/heuristic scores; class shares are predicted pixel shares. |
| 3:40-4:20 | Export All results ZIP; show its contents | Images, labels, arrays and reports; GeoTIFF retains CRS and footprint. |
| 4:20-5:00 | Select water or vegetation sample | Repeated image selection works; describe real-data training as the next validation step. |

## A short presentation explanation

“Our project implements a complete four-band satellite processing pipeline. RCAN learns 2x reconstruction. We reproduce a synthetic training experiment and integrate a real-pretrained OpenEarthMap land-cover detector. It detects from original imagery, with measured-NIR consistency checks for multispectral scenes. We demonstrate on genuine Sentinel-2 data and test annotated high-resolution satellite examples. The app supports CPU laptops and exports both visual and GIS outputs. Synthetic evaluation shows a reconstruction improvement over bicubic. Real-scene class accuracy and true resolution gains require independent ground truth and remain future validation work.”

## Questions you may receive

**Does it find buildings, roads, trees and water?** It predicts building, road, vegetation, water and bare-land pixels. Vegetation includes trees; individual objects are not counted. The real-pretrained detector can still make errors because of domain shift and limited input resolution. Developed space remains separate from roads/buildings.

**Does it increase image quality?** It produces learned 2x reconstruction and a sharper appearance. Reconstruction quality improved on held-out synthetic ground truth. True reconstruction quality on these real scenes is not measured.

**Is 98.13% the accuracy on Delhi?** No. It is held-out synthetic pipeline accuracy. Real-scene accuracy is unknown.

**Can it work without a GPU or internet?** Yes after dependency installation. CPU executes both trained models; included samples/checkpoints and prepared outputs are available offline.

**What was tested?** All 78 checks passed in the CUDA-capable installation; all 76 applicable checks passed in the CPU-only build, with two expected GPU skips. Full results are recorded in real_detector_audit.json. They cover file formats, invalid uploads, inference parity, memory, repeated selection, exports, training and all five real samples. These are software checks, not labeled real-scene accuracy measurements.

**What belongs in the submission?** The ZIP contains source code, tests, dependency lists, trained checkpoints, ONNX model, real data/provenance, current reports, demo instructions and prepared real-scene results. Local environments, unrelated documents and the older unreviewed presentation are excluded. Use PROJECT_REPORT.md and this guide for current claims.
