# Detic ROI visibility test package

This package lets you test FIction-Detic / Detic as a drop-in detector backend for your current ROI-based visibility pipeline.

## Files

- `common.py`  
  Shared dataclasses: `ROIBox`, `DetectionResult`, `VisibilityResult`.

- `roi_visibility.py`  
  Detector-agnostic ROI visibility estimator. It builds the projected ROI, runs a detector inside the ROI, maps detections back to full-frame coordinates, scores confidence + location + size, and draws debug output.

- `detic_detector.py`  
  Thin wrapper around FIction-Detic / facebookresearch Detic. It returns the same `DetectionResult` format as your Grounding DINO wrapper.

- `test_detic_roi.py`  
  One-image test script.

## Setup

Detic depends on Detectron2 and CenterNet2. It is easiest to test in Linux, WSL2, or Colab with CUDA.

Clone FIction-Detic:

```bash
git clone https://github.com/thechargedneutron/FIction-Detic.git
cd FIction-Detic
```

Follow the repo's install instructions, then download the weight:

```bash
mkdir -p models
wget https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth \
  -O models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
```

## Example: your current test frame

```bash
python test_detic_roi.py \
  --image "/Users/fangzhouma/Desktop/3d_vision/3D-VLM-benchmark-OOS/data/for_debug/time_6.jpg" \
  --detic-root "/path/to/FIction-Detic" \
  --u 408.3171177561637 \
  --v 528.5396005453204 \
  --prompt pot pan saucepan cooking-pot \
  --expected-width 120 \
  --expected-height 120 \
  --last-seen-bbox 584.86154 568.01368 909.78462 758.15385 \
  --box-threshold 0.25 \
  --visible-threshold 0.35 \
  --partial-threshold 0.18 \
  --roi-scale 1.8 \
  --output detic_roi_debug.jpg \
  --json-output detic_roi_result.json
```

## Notes

For custom vocabulary mode, prefer short category names:

Good:

```bash
--prompt pot pan saucepan cooking-pot
```

Riskier:

```bash
--prompt "a pot in the fridge"
```

Detic is a category detector, not a full phrase-grounding model. The ROI/projection handles the spatial part, while Detic handles the object category recognition.
