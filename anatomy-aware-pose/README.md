# Anatomy-Aware Pose Estimation on Lightweight Networks

**Computer Vision** — Prof. Irene Amerini, Sapienza Università di Roma, Spring 2026

> Human pose estimation with anatomical constraints (Skeletal Topology Loss)
> on a lightweight MobileNetV3 backbone, targeting real-time edge deployment
> for Human-Robot Interaction safety.

---

## Table of Contents

1. [Problem and Motivation](#1-problem-and-motivation)
2. [Our Contribution](#2-our-contribution)
3. [Code Structure](#3-code-structure)
4. [Datasets](#4-datasets)
5. [Data Pipeline — Step by Step](#5-data-pipeline--step-by-step)
6. [Network Architecture](#6-network-architecture)
7. [Training](#7-training)
8. [Evaluation Pipeline — Step by Step](#8-evaluation-pipeline--step-by-step)
9. [Anatomical Violation Rate (AVR)](#9-anatomical-violation-rate-avr)
10. [Still To Implement](#10-still-to-implement)
11. [How to Run](#11-how-to-run)
12. [Reproducibility](#12-reproducibility)
13. [References](#13-references)

---

## 1. Problem and Motivation

Standard human pose estimation (HPE) models predict 2D joint locations from
images. Under **occlusion** — when body parts are hidden behind objects or other
people — these models often produce predictions that are **anatomically
impossible**: an elbow at an impossible distance from the shoulder, a knee
bending backwards, limbs collapsing into a single point.

For a typical computer vision benchmark this is just a lower AP score. But for
a **humanoid robot** operating in Human-Robot Interaction (HRI), an anatomically
impossible pose is a **safety hazard**: the robot's motion planner uses the
estimated human pose to plan collision-free trajectories. If the pose is
physically impossible, the planner may allow the robot to move into space that
is actually occupied by the human.

There is a second constraint: HRI happens on **edge hardware** (e.g. NVIDIA
Jetson Orin on a Unitree H1 humanoid). State-of-the-art models like ViTPose
(~632M params) or HRNet (~63M params) are too heavy for real-time inference on
these devices.

**We need a model that is both lightweight AND anatomically aware.**

---

## 2. Our Contribution

We propose a three-part contribution:

### 2.1 Skeletal Topology Loss (STL)
A differentiable, plug-in loss function with three terms:

| Term | What it penalizes | Example |
|------|-------------------|---------|
| **Bone ratio** | Limb-length ratios outside anthropometric ranges | Forearm/upper-arm ratio outside [0.71, 0.85] |
| **Joint angle** | Joint angles outside physiological ranges | Elbow angle < 5° or > 175° |
| **Geometric ordering** | Joints out of order along kinematic chains | Knee not between hip and ankle |

The constraints are **absolute and anthropometric** (from biomechanical
literature), NOT statistical values learned from the dataset. This is the key
differentiator from Han et al. (2025), who learn their constraints from the
training data distribution.

### 2.2 Lightweight backbone
MobileNetV3-Small (~2.5M params) with a deconvolution head, chosen for
real-time edge deployment. We frame the expected AP gap vs. heavier models as
a latency/FLOPs tradeoff, not a weakness.

### 2.3 Custom metric: Anatomical Violation Rate (AVR)
A safety-oriented metric that counts the percentage of predicted poses
violating anatomical constraints. Complementary to AP/AR: even if AP improves
only marginally, a significant drop in AVR at the same AP level is a
publishable result for HRI.

### 2.4 Evaluation protocol
Train on **MS-COCO Keypoints**, evaluate **zero-shot on OCHuman** (no
fine-tuning). This cross-dataset protocol isolates whether the model has truly
learned anatomical priors that generalize, rather than just memorizing
dataset-specific patterns.

---

## 3. Code Structure

The code follows the structure required by the professor:
**Imports → Globals → Utils → Data → Network → Train → Evaluation**

```
├── config.py              # Globals: paths, hyperparameters, seed, device
├── utils.py               # Utils: heatmap generation/decoding, coordinate transforms
├── data.py                # Data: COCO/OCHuman parsing, Dataset classes
├── network.py             # Network: MobileNetV3 + DeconvHead
├── train.py               # Train: WeightedMSELoss, training loop, checkpointing
├── evaluation.py          # Evaluation: inference, AP/AR, AVR, COCO/OCHuman wrappers
├── kaggle_runner.ipynb    # Minimal notebook that clones the repo and runs everything
├── .gitignore             # Excludes datasets, checkpoints, __pycache__
└── README.md              # This file
```

**Workflow:** GitHub is the single source of truth for code. Kaggle is only the
training/evaluation machine. The runner notebook clones the repo at the start
of every session, so it always uses the latest version. Datasets and
checkpoints stay on Kaggle (never on Git — they are too large).

---

## 4. Datasets

### MS-COCO 2017 Keypoints (training + validation)
- **116,021** training examples (persons with ≥8 annotated keypoints, bbox area ≥ 32×32)
- **6,352** validation examples
- 17 keypoints per person in COCO order:
  `nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder,
   left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip,
   left_knee, right_knee, left_ankle, right_ankle`
- Keypoint visibility: `v=0` not annotated, `v=1` annotated but occluded, `v=2` visible
- Kaggle dataset: `asad11914/coco-2017-keypoints`

### OCHuman (zero-shot evaluation)
- **4,731 images, 8,110 persons** (standard subset with keypoint+mask annotations)
- High occlusion by design (~40-80% IoU between person bounding boxes)
- COCO-style annotation format → same code works for both datasets
- Uploaded as shared Kaggle dataset (academic license from Tsinghua)
- ⚠️ Images are double-nested: `.../ochuman/images/images/`

---

## 5. Data Pipeline — Step by Step

This section explains what happens to a single training example, from raw COCO
image to the tensors that enter the network. Understanding this chain is
essential because the **evaluation pipeline must invert it exactly** to map
predictions back to the original image space.

### 5.1 Sample construction (`data.build_samples`)

We iterate over all COCO annotations. For each person annotation we extract:
- `image_id`, `file_name` (to load the image later)
- `bbox` = [x, y, w, h] (bounding box in original image coordinates)
- `keypoints` = 51 values (17 keypoints × 3: x, y, visibility)

Filtering: we discard persons with fewer than 8 annotated keypoints (too few
to learn from) or with degenerate bounding boxes (area < 32×32).

### 5.2 Aspect-ratio preserving resize (`data._crop_and_pad`)

This is an **architecturally critical** design choice. The standard approach in
many codebases is to stretch the bbox crop to the target size (256×192). We
deliberately use **uniform scaling + centered padding** instead:

```
Original bbox (say 200×400, tall person)
    │
    ▼  scale = min(192/200, 256/400) = min(0.96, 0.64) = 0.64
Scaled to 128×256 (same aspect ratio)
    │
    ▼  pad_left = (192 - 128) / 2 = 32 px of black padding on each side
Canvas 256×192 with person centered, black bars on sides
```

**Why not stretch?** Because the STL reasons about bone-length *ratios* and
joint *angles*. A non-uniform stretch (e.g. squishing a tall person
horizontally) would distort these proportions: a bone ratio of 0.75 in reality
might become 0.60 after stretch. The loss would penalize a correct pose or,
worse, accept an incorrect one. Uniform scaling preserves all geometric
relationships. This choice must be defended in the paper and during the exam.

### 5.3 Keypoint coordinate transform

Each keypoint must be transformed from original image space to heatmap space,
following the same chain:

```
Original (kx, ky) in full image
    │  subtract bbox origin
    ▼
(kx - x, ky - y)    in bbox-local coordinates
    │  apply same scale factor
    ▼
(kx - x) * scale + pad_left    in canvas (256×192) coordinates
    │  scale to heatmap resolution (÷4)
    ▼
cx = ((kx - x) * scale + pad_left) * (48 / 192)    in heatmap (64×48)
cy = ((ky - y) * scale + pad_top)  * (64 / 256)    in heatmap (64×48)
```

If `v=0` (keypoint not annotated): heatmap is all zeros, `target_weight = 0`
(the loss ignores this keypoint entirely).

### 5.4 Gaussian heatmap generation (`utils.generate_heatmap`)

For each visible keypoint, we place a 2D Gaussian centered at `(cx, cy)`:

```
heatmap[y, x] = exp( -((x - cx)² + (y - cy)²) / (2σ²) )
```

with `σ = 2`. The peak is 1.0 exactly at the center; it falls off smoothly.
This is the **ground truth** the model learns to predict: one heatmap per
keypoint, 17 heatmaps total, each 64×48.

### 5.5 Output tensors

Each training example produces:
- `img_tensor` — shape `[3, 256, 192]`, pixel values in `[0, 1]`
- `hm_tensor` — shape `[17, 64, 48]`, ground truth Gaussian heatmaps
- `weight_tensor` — shape `[17, 1]`, 1.0 for valid keypoints, 0.0 for missing

Batched with `DataLoader(batch_size=32)`:
`[32, 3, 256, 192]`, `[32, 17, 64, 48]`, `[32, 17, 1]`.

---

## 6. Network Architecture

### 6.1 Backbone: MobileNetV3-Small

We use `torchvision.models.mobilenet_v3_small` pretrained on ImageNet. We take
only the `.features` part (the convolutional trunk), discarding the classifier.

Input `[B, 3, 256, 192]` → features `[B, 576, 8, 6]` (stride 32×).

~1.5M parameters. Efficient on edge hardware due to depthwise separable
convolutions and squeeze-and-excitation blocks.

### 6.2 Head: Deconvolution (transposed convolutions)

Three `ConvTranspose2d` layers, each with stride 2 (= upsample 2×), followed
by BatchNorm and ReLU. Total upsample: 2³ = 8×. A final 1×1 Conv maps to 17
channels (one per keypoint).

```
[B, 576, 8, 6]
    │  ConvTranspose2d(576→256, k=4, s=2, p=1) + BN + ReLU
    ▼
[B, 256, 16, 12]
    │  ConvTranspose2d(256→256, k=4, s=2, p=1) + BN + ReLU
    ▼
[B, 256, 32, 24]
    │  ConvTranspose2d(256→256, k=4, s=2, p=1) + BN + ReLU
    ▼
[B, 256, 64, 48]
    │  Conv2d(256→17, k=1)
    ▼
[B, 17, 64, 48]   ← predicted heatmaps
```

~3M parameters in the head. Total model: ~4.5M parameters.

**Note:** the backbone has stride 32 (256/8), and the head upsamples 8× back
to the heatmap resolution (64×48). The overall stride from input to heatmap is
4× (256/64 = 192/48 = 4), which is standard in pose estimation.

---

## 7. Training

### 7.1 Loss: WeightedMSELoss (`train.py`)

Per-pixel MSE between predicted and ground-truth heatmaps, with per-keypoint
weighting:

```python
loss = MSE(pred, target)          # [B, K, H*W]  per-pixel squared error
     .mean(dim=-1)                # [B, K]        average over spatial dims
     * target_weight.squeeze(-1)  # [B, K]        zero out missing keypoints
     .sum()                       # scalar
     / (target_weight.sum() + ε)  # normalize by number of valid keypoints
```

The `+ ε` (1e-6) prevents division by zero if all keypoints in a batch happen
to be invalid. The `target_weight` comes from the dataset: it is 0.0 for
keypoints with `v=0` (not annotated) and 1.0 otherwise.

**Why MSE on heatmaps, not L1 or cross-entropy?** MSE is the standard in
heatmap-based pose estimation (SimpleBaseline, HRNet, ViTPose all use it). The
Gaussian shape of the target means MSE naturally penalizes predictions near the
peak more than those far away.

### 7.2 Optimizer and scheduler

- **Adam** with initial lr = 1e-3
- **MultiStepLR**: lr drops ×0.1 at epoch 15 and 25
- 30 epochs total

### 7.3 Checkpointing and resume (`train.fit`)

Every epoch saves:
- `last.pth` — full state (model + optimizer + scheduler + epoch + val_loss),
  for resuming interrupted training
- `best.pth` — model weights only, updated when val_loss improves

On startup, `fit()` checks if `last.pth` exists. If so, it reloads everything
and resumes from the next epoch. This is critical on Kaggle where sessions can
die mid-training.

A `history.csv` is written at the end with per-epoch metrics.

---

## 8. Evaluation Pipeline — Step by Step

The evaluation pipeline lives in `evaluation.py`. It has three layers:

```
evaluate_on_coco_val / evaluate_on_ochuman    ← high-level wrappers
    │
    ├── run_inference         ← model forward pass on all samples
    │       ├── decode_heatmaps         (heatmap → coordinates in heatmap space)
    │       └── heatmap_to_original     (heatmap space → original image space)
    │
    ├── run_coco_eval         ← AP/AR via pycocotools (needs ground truth)
    │
    └── evaluate_avr          ← AVR custom metric (does NOT need ground truth)
```

### 8.1 Inference: from image to keypoint coordinates

#### `COCOEvalDataset` (data.py)

Unlike the training dataset, this returns **(image, image_id, bbox)** — no
ground truth heatmaps. The ground truth enters separately via the annotation
file. The image goes through the **same** `_crop_and_pad` as training (same
scale, same padding), ensuring geometric consistency.

The `bbox` is returned alongside the image because we need it later to invert
the coordinate transform.

#### `run_inference` (evaluation.py)

For each batch:
1. Forward pass → predicted heatmaps `[B, 17, 64, 48]`
2. `decode_heatmaps` → coordinates in heatmap space + confidence scores
3. `heatmap_to_original` → coordinates in original image space
4. Pack into COCO results format (list of dicts with `image_id`, `category_id`,
   `keypoints` as flat [x1,y1,s1, x2,y2,s2, ...], `score`)

### 8.2 Heatmap decoding (`utils.decode_heatmaps`)

**Input:** predicted heatmaps `[B, K, H, W]` (raw model output, NOT softmaxed)

**Step 1 — argmax:** flatten each heatmap to 1D, find the index of the maximum
value. Convert that flat index back to (x, y) coordinates:
```python
x = idx % W      # column
y = idx // W      # row
```
The value at the peak becomes the **confidence score** for that keypoint.

**Step 2 — sub-pixel refinement:** the argmax gives integer coordinates, but the
true peak is almost never exactly on a grid point. We shift by ±0.25 pixels
in the direction of the local gradient:
```python
dx = heatmap[y, x+1] - heatmap[y, x-1]    # horizontal gradient
dy = heatmap[y+1, x] - heatmap[y-1, x]    # vertical gradient
x += 0.25 * sign(dx)
y += 0.25 * sign(dy)
```
This is a standard trick from SimpleBaseline (Xiao et al., ECCV 2018). It
improves AP by ~0.5-1% at zero computational cost. The guard `1 < x < W-1`
avoids out-of-bounds access at the heatmap borders.

**Output:** coordinates `[B, K, 2]` in heatmap space + scores `[B, K]`.

### 8.3 Coordinate inverse transform (`utils.heatmap_to_original`)

This **inverts** the chain from Section 5 to go from heatmap space back to
original image coordinates. The chain to invert is:

```
original → crop (subtract bbox origin) → scale → pad → heatmap (÷4)
```

So the inverse is:

```
heatmap coordinates (e.g. cx=20.25, cy=35.75 in 64×48 space)
    │  ×(input_w / hm_w) and ×(input_h / hm_h)     ← undo ÷4
    ▼
canvas coordinates (in 256×192 space)
    │  subtract pad_left, pad_top                    ← undo padding
    │  divide by scale                               ← undo uniform scaling
    │  add bbox origin (x, y)                        ← undo crop
    ▼
original image coordinates (in full image pixel space)
```

The parameters (scale, pad_left, pad_top) are **recomputed** from the bbox
using the same formula as `_crop_and_pad`, ensuring perfect consistency. This
is why `COCOEvalDataset` returns the bbox alongside the image.

### 8.4 AP / AR evaluation (`evaluation.run_coco_eval`)

Uses the official **pycocotools** COCO evaluation protocol:

1. Save predictions as a JSON file in COCO results format
2. Load ground truth annotations with `COCO(ann_file)`
3. Load predictions with `coco_gt.loadRes(results_path)`
4. Create `COCOeval` with `iouType='keypoints'`
5. `evaluate()` → `accumulate()` → `summarize()`

The metric is **OKS** (Object Keypoint Similarity), which is the keypoint
equivalent of IoU. It measures how close each predicted keypoint is to the
ground truth, normalized by the person's scale (bbox area) and a per-keypoint
constant (σ) that accounts for annotation noise (e.g. hip is harder to
annotate precisely than nose).

`summarize()` prints 10 standard metrics:
AP, AP@.50, AP@.75, AP_M (medium), AP_L (large),
AR, AR@.50, AR@.75, AR_M, AR_L.

**Note:** we evaluate using **ground-truth bounding boxes** (top-down protocol
with GT boxes, not a person detector). This is standard for ablation studies
and isolates pose estimation quality from detection quality. Must be documented
in the paper as "GT bbox protocol".

### 8.5 High-level wrappers

#### `evaluate_on_coco_val(model, val_samples, device)`

Runs the full pipeline on COCO validation:
`run_inference` → `run_coco_eval` (AP/AR against COCO GT) → `evaluate_avr`.
Returns `(coco_eval, avr_dict)`.

#### `evaluate_on_ochuman(model, device)`

Same pipeline but on OCHuman. **Zero-shot** means the model has never seen
OCHuman images during training — no fine-tuning. This tests whether anatomical
priors generalize across datasets.

OCHuman is in COCO-style format, so the same `build_samples` + `run_inference`
+ `run_coco_eval` code works unchanged. AP/AR use OCHuman's GT keypoints;
AVR does not use GT at all.

---

## 9. Anatomical Violation Rate (AVR)

The AVR is our **custom safety metric**. Unlike AP/AR, it does NOT compare
predictions against ground truth. Instead, it checks whether the predicted
pose is **internally consistent** with human anatomy: are the bone lengths
plausible? Are the joint angles physically possible? Are joints collapsing
into each other?

This is important for HRI because a pose can have decent AP (joints roughly in
the right place) but still be anatomically impossible — which would confuse a
robot's motion planner.

### 9.1 Confidence gating

Before checking any constraint, we verify that the relevant keypoints are
**confident enough** (score ≥ 0.3). A keypoint the model is unsure about
should not trigger a violation — the model is effectively saying "I don't know
where this joint is", which is honest, not a violation.

### 9.2 Three types of violations

#### Bone ratio violations (`bone_ratio`)

We check 4 symmetric bone pairs (left vs right):
- Upper arm: shoulder→elbow (left vs right)
- Forearm: elbow→wrist (left vs right)
- Thigh: hip→knee (left vs right)
- Lower leg: knee→ankle (left vs right)

For each pair, we compute `max(left_len, right_len) / min(left_len, right_len)`.
If this ratio exceeds **1.5**, it is a violation — one side is 50% longer than
the other, which is anatomically impossible.

**Note:** this current implementation checks **left/right symmetry**, not
absolute anthropometric ratios (e.g. forearm/upper-arm should be in
[0.71, 0.85]). The absolute ratio constraints are part of the **STL**
(still to implement) and will use citable biomechanical sources.

#### Joint angle violations (`joint_angle`)

We check 4 major joints:
- Left/right knee (hip→knee→ankle)
- Left/right elbow (shoulder→elbow→wrist)

The angle at the joint vertex is computed via the dot product formula:
```
cos(θ) = (v1 · v2) / (|v1| × |v2|)
θ = arccos(clip(cos(θ), -1, 1))
```

If `θ < 20°`, it is a violation — the joint is almost completely folded, which
typically indicates a prediction error (two joints collapsing to the same
location) rather than a real pose.

#### Collapse violations (`collapse`)

A joint is "collapsed" onto one of its neighbors if the distance between them,
**normalized by torso length** (for scale invariance), is below 10% of torso
length. The torso is measured as the distance between the midpoint of the
shoulders and the midpoint of the hips.

This catches a failure mode where the model predicts several joints at nearly
the same pixel location — geometrically valid in terms of angles but
anatomically meaningless.

### 9.3 Aggregation

- `AVR_pose_rate`: fraction of poses with **at least one** violation (any type)
- `AVR_mean_violations`: average number of violations per pose
- `per_category`: breakdown by type (bone_ratio, joint_angle, collapse)

**Expected baseline behavior:** the baseline (no STL) should have a
non-negligible AVR, especially on OCHuman where occlusion is heavy. After
adding the STL, the AVR should drop — this is the main result we are after.

---

## 10. Still To Implement

### 10.1 Skeletal Topology Loss (STL) — the core contribution

The STL is a differentiable multi-term loss that replaces the AVR's hard-coded
violation checks with smooth, gradient-friendly penalties that the model can
learn from during training.

**Critical prerequisite: soft-argmax.** The STL operates on keypoint
*coordinates*, but the model outputs *heatmaps*. The `argmax` used in
evaluation is not differentiable, so we need a differentiable alternative:

```python
def soft_argmax(heatmaps):
    """[B, K, H, W] → [B, K, 2] differentiable coordinates."""
    B, K, H, W = heatmaps.shape
    softmax = F.softmax(heatmaps.view(B, K, -1), dim=-1).view(B, K, H, W)
    grid_x = torch.arange(W, device=heatmaps.device).float()
    grid_y = torch.arange(H, device=heatmaps.device).float()
    x = (softmax.sum(dim=2) * grid_x).sum(dim=-1)  # [B, K]
    y = (softmax.sum(dim=3) * grid_y).sum(dim=-1)  # [B, K]
    return torch.stack([x, y], dim=-1)              # [B, K, 2]
```

**Three terms to implement:**

1. **Bone ratio term** — for each anatomical bone (defined as pairs of
   keypoints), compute the predicted length, divide by a reference length
   (e.g. torso), and penalize if the ratio falls outside the anthropometric
   range from biomechanical literature.
   - Sources needed: Winter (2009) "Biomechanics and Motor Control of Human
     Movement", Drillis & Contini (1966) segment ratios.
   - Must be **citable** — this is the claim that differentiates us from Han
     et al. (who learn their ranges from data).

2. **Joint angle term** — compute the angle at each joint via the
   differentiable `atan2`-based formula (NOT the arccos used in AVR, which has
   gradient issues at 0° and 180°). Penalize if outside the physiological
   range from biomechanical literature.

3. **Geometric ordering term** — model the skeleton as a kinematic tree.
   For each chain (e.g. hip→knee→ankle), ensure the intermediate joint lies
   *between* the extremes along the chain direction. Implemented as a soft
   penalty on the projection.

**Combined loss:**
```
L_total = L_heatmap + λ_bone × L_bone + λ_angle × L_angle + λ_order × L_order
```

The λ weights need tuning via grid search on a validation subset.

**Differentiability test:** every term must pass `torch.autograd.gradcheck` on
a small synthetic input before being used in training.

### 10.2 Grad-CAM explainability

Grad-CAM on the last convolutional layer of the backbone, targeting specific
keypoint channels. The goal: verify that when a keypoint is occluded, the
model looks at *neighboring visible joints* to infer its position (evidence
that the STL is teaching anatomical reasoning), rather than hallucinating.

Compare Grad-CAM visualizations between baseline (no STL) and STL model on
the same occluded examples.

### 10.3 Ablation study

Required by the project specification. Four configurations:
1. Heatmap loss only (baseline) — already training
2. Heatmap + bone ratio term
3. Heatmap + bone ratio + joint angle
4. Full STL (all three terms)

For each: AP, AR, AVR on both COCO val and OCHuman zero-shot. Plus λ
sensitivity analysis (how AP/AVR trade off as λ increases).

### 10.4 Inference benchmark

Measure inference time (ms/image) on CPU and GPU. Compare with ViTPose and
HRNet. Count FLOPs with `thop` or `fvcore`. This justifies the lightweight
backbone choice.

### 10.5 Paper and presentation

- ICRA/IROS format: Introduction, Related Work, Methodology (STL equations),
  Experiments, Results, Conclusions
- 15-20 slides for the exam
- Failure case analysis: side-by-side baseline vs STL on occluded poses

---

## 11. How to Run

### On Kaggle (training + evaluation)

1. Upload `kaggle_runner.ipynb` (File → Import Notebook)
2. Settings → **Internet: On**
3. **Add Input**: `COCO 2017 Keypoints` + shared `OCHuman` dataset
4. In cell 1, set `REPO_URL` to this repository
5. Run All. Training saves `best.pth`; evaluation prints AP/AR/AVR table.

For overnight training: **Save Version** → Save & Run All (Commit). Runs in
background; results appear in the Version's Output tab.

### On local machine (editing only — no training)

```bash
git clone https://github.com/YOUR_USER/YOUR_REPO.git
cd YOUR_REPO
# edit .py files
git add . && git commit -m "description" && git push
# on Kaggle: the runner re-clones automatically
```

### Working in two

- One edits locally, `git pull` before starting, `git push` when done.
- The Kaggle runner re-clones at every execution → always gets the latest code.
- Never edit code directly on Kaggle — always edit locally and push.

---

## 12. Reproducibility

- Seed: `SEED = 42`, applied to `random`, `numpy`, `torch`, `torch.cuda`,
  `cudnn.deterministic = True`, `cudnn.benchmark = False`
- Framework: Python 3.10, PyTorch (Kaggle default), torchvision
- GPU: NVIDIA T4 (Kaggle) — note that `num_workers > 0` introduces minor
  non-determinism in batch ordering across runs
- All hyperparameters in `config.py`
- Training history logged to `history.csv`

---

## 13. References

1. **ViTPose** — Xu et al., "ViTPose: Simple Vision Transformer Baselines for
   Human Pose Estimation", NeurIPS 2022
2. **Han et al.** — "Human pose estimation method based on limb graph
   structure", Neural Computing & Applications, 2025
   (DOI: 10.1007/s00521-024-10676-3)
3. **SimCC / Liu et al.** — DSP 2024 (bone-length loss without angle
   constraints — gap we fill)
4. **Pytel et al.** — "Tilting at windmills: Data augmentation for deep pose
   estimation does not help with occlusion", ICPR 2021
5. **Pose2Seg / Zhang et al.** — "Pose2Seg: Detection Free Human Instance
   Segmentation", CVPR 2019 (OCHuman dataset)
6. **SimpleBaseline** — Xiao et al., "Simple Baselines for Human Pose
   Estimation and Tracking", ECCV 2018 (deconv head design, sub-pixel
   refinement)

### Biomechanical sources (to be added for STL)
- Winter, D.A. "Biomechanics and Motor Control of Human Movement", 4th ed., Wiley, 2009
- Drillis, R. & Contini, R. "Body Segment Parameters", Technical Report, NYU, 1966
