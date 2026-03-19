

# Prompted Segmentation for Drywall QA

> Fine-tuned text-conditioned segmentation model that takes an image and a natural-language prompt and returns a pixel-level binary mask for construction defects.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9.0+cu126-orange)
![Transformers](https://img.shields.io/badge/Transformers-5.2.0-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Demo

| Prompt | Input | Prediction |
|---|---|---|
| `segment crack` | Wall image | White pixels = crack location |
| `segment taping area` | Drywall image | White pixels = joint/tape location |

---

## Overview

This project fine-tunes **CLIPSeg** (a vision-language segmentation model) on two construction defect datasets so that given any wall/drywall image and a text prompt, the model outputs a binary mask highlighting the defect region.

### Supported Prompts

**Cracks:**
- `segment crack`
- `segment wall crack`
- `find crack`
- `highlight crack`

**Taping / Joints:**
- `segment taping area`
- `segment joint/tape`
- `segment drywall seam`
- `segment drywall joint`

---

## Model Architecture
```
Text prompt  →  CLIP Text Encoder   ─┐
                                      ├→  Cross-attention decoder  →  1×1 conv  →  Bilinear upsample  →  Binary mask
Image        →  CLIP Vision Encoder  ─┘
```

**Base model:** [CIDAS/clipseg-rd64-refined](https://huggingface.co/CIDAS/clipseg-rd64-refined)  
**Fine-tuned on:** Drywall joints + wall cracks (6100 training images)  
**Output:** Single-channel PNG, same size as input, values `{0, 255}`

---

## Results

| Prompt | mIoU ↑ | Dice ↑ |
|---|---|---|
| segment taping area | 0.5463 | 0.6959 |
| segment crack | 0.5200 | 0.6657 |
| **Average** | **0.5332** | **0.6808** |

### Training Dynamics

| Epoch | Train Loss | Val mIoU | Val Dice |
|---|---|---|---|
| 5  | ~0.42 | ~0.38 | ~0.52 |
| 10 | ~0.35 | ~0.44 | ~0.58 |
| 20 | ~0.29 | ~0.50 | ~0.64 |
| 30 | 0.2575 | 0.5345 | 0.6826 |

---

## Datasets

| Dataset | Category | Train | Valid | Annotation type |
|---|---|---|---|---|
| [Drywall-Join-Detect](https://universe.roboflow.com/objectdetect-pu6rn/drywall-join-detect) | taping | 936 | 250 | Bounding box → filled rect mask |
| [Cracks-3ii36](https://universe.roboflow.com/fyp-ny1jt/cracks-3ii36) | crack | 5164 | 201 | Polygon → binary mask |
| **Combined** | both | **6100** | **451** | — |

---

## Environment

| Package | Version |
|---|---|
| Python | 3.12 |
| torch | 2.9.0+cu126 |
| transformers | 5.2.0 |
| albumentations | 2.0.8 |
| numpy | 2.0.2 |
| GPU | Tesla T4 (Kaggle) |

---

## Quick Start

### Installation
```bash
pip install torch torchvision transformers albumentations Pillow numpy tqdm
```


# ── Run 
mask = predict("your_wall_image.jpg", "segment crack")
Image.fromarray(mask, mode="L").save("output__segment_crack.png")
print("Mask saved!")

## Output Format
# Single-channel PNG
# Values: 0 = background, 255 = defect detected
# Filename convention:
{image_stem}__{prompt_slug}.png

# Examples:
img_0042__segment_crack.png
img_0107__segment_taping_area.png
```

---

## Training Details

| Parameter | Value |
|---|---|
| Seed | 42 |
| Epochs | 30 |
| Batch size | 8 |
| Optimizer | AdamW |
| Backbone LR | 3e-6 (0.1× head LR) |
| Head LR | 3e-5 |
| Scheduler | CosineAnnealingLR (T_max=30, eta_min=1e-7) |
| Loss | 0.5 × BCE + 0.5 × Dice |
| Precision | FP16 (torch.amp autocast) |
| Grad clipping | max_norm=1.0 |
| Training time | ~2 hours on Tesla T4 |
| Model size | 575 MB |

```

## Failure Notes

- **Hairline cracks (<3px)** — CLIPSeg's 352px internal resolution misses sub-pixel cracks
- **Bbox annotations for taping** — dataset had no polygon masks, bounding boxes introduce boundary noise at region edges
- **Class imbalance** — taping covers 20–40% of pixels, cracks cover 2–5%, single model must balance both
- **Overexposed images** — high-key whites reduce crack contrast, ~35% of cracks missed in this subset

---

## Reproducibility

All random seeds pinned to `42`:
```python
import random, numpy as np, torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
```

---

## Kaggle Notebook

[https://www.kaggle.com/code/darpan5678/drywall-qa-final](https://www.kaggle.com/code/darpan5678/drywall-qa-final)

---

## References

- Lüddecke, T., & Ecker, A. (2022). [Image Segmentation Using Text and Image Prompts](https://arxiv.org/abs/2112.10003). CVPR 2022.
- [CLIPSeg HuggingFace Model](https://huggingface.co/CIDAS/clipseg-rd64-refined)
- [Roboflow Drywall Dataset](https://universe.roboflow.com/objectdetect-pu6rn/drywall-join-detect)
- [Roboflow Cracks Dataset](https://universe.roboflow.com/fyp-ny1jt/cracks-3ii36)