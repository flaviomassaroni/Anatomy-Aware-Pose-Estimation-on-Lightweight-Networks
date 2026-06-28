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
10. [Baseline Results](#10-baseline-results)
11. [Skeletal Topology Loss (STL) — Implementation](#11-skeletal-topology-loss-stl--implementation)
12. [Still To Do](#12-still-to-do)
13. [How to Run](#13-how-to-run)
14. [Reproducibility](#14-reproducibility)
15. [References](#15-references)

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
A differentiable, plug-in loss function with four terms:

| Term | What it penalizes | Example |
|------|-------------------|---------|
| **Bone ratio** | Limb-length ratios outside anthropometric ranges | Forearm/upper-arm ratio deviating from Winter 2009 nominal |
| **Joint angle** | Joint angles outside physiological ranges | Elbow or knee angle < 20° |
| **Geometric ordering** | Joints out of order along kinematic chains | Knee not between hip and ankle |
| **Bone collapse** | Segments shorter than 10% of torso length | Elbow collapsed onto shoulder |

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
├── config.py                        # Globals: paths, hyperparameters, seed, device, STL/AVR thresholds
├── utils.py                         # Utils: heatmap generation/decoding, coordinate transforms
├── data.py                          # Data: COCO/OCHuman parsing, Dataset classes
├── network.py                       # Network: MobileNetV3 + DeconvHead
├── train.py                         # Train: WeightedMSELoss, training loop, checkpointing
├── evaluation.py                    # Evaluation: inference, AP/AR, AVR, COCO/OCHuman wrappers
├── anthropometric_constraints.py    # Anthropometric ranges from biomechanical literature
├── stl.py                           # Skeletal Topology Loss (soft-argmax + 4 differentiable terms)
├── test_stl.py                      # Differentiability tests (gradcheck) for all STL components
├── kaggle_runner.ipynb              # Minimal notebook that clones the repo and runs everything
├── .gitignore
└── README.md
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
[0.55, 1.05]). The absolute ratio constraints are in the **STL** (see
Section 10) and use citable biomechanical sources (Winter 2009, Drillis
& Contini 1966).

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

**Baseline behavior (measured):** the baseline (no STL) has AVR = 30.6% on
COCO val and AVR = 48.2% on OCHuman — confirming that occlusion drives
anatomical violations. The goal of the STL is to lower this AVR **without**
destroying AP. Our first attempt at STL fine-tuning actually made both worse;
the diagnosis and the fixes that followed are documented in Section 11.9.

---

## 10. Baseline Results

Training: 30 epochs, MobileNetV3-Small, heatmap MSE loss, Adam lr=1e-3 with
MultiStepLR (drops ×0.1 at epochs 15, 25). Best val_loss at epoch 16.

| Metric | COCO val | OCHuman (zero-shot) |
|--------|----------|---------------------|
| AP     | 0.498    | 0.436               |
| AP@.50 | 0.777    | 0.779               |
| AP@.75 | 0.532    | 0.433               |
| AR     | 0.538    | 0.523               |
| **AVR rate** | **0.306** | **0.482**      |
| **AVR mean** | **0.405** | **0.694**      |

Key observations:
- AP@.50 is nearly identical between COCO (0.777) and OCHuman (0.779) — the
  keypoints are roughly in the right place even under heavy occlusion.
- But the **AVR doubles** from 30.6% to 48.2% — almost half of all predicted
  poses on OCHuman have at least one anatomical violation.
- This confirms the core thesis: AP alone does not capture anatomical
  plausibility. A robot trusting these poses would plan unsafe trajectories.

---

## 11. Skeletal Topology Loss (STL) — Implementation

The STL is implemented in `stl.py` with anthropometric ranges in
`anthropometric_constraints.py` and shared thresholds in `config.py`.
All terms pass `torch.autograd.gradcheck` (verified in `test_stl.py`).

### 11.1 Soft-Argmax — the bridge between heatmaps and coordinates

The model outputs heatmaps `[B, 17, 64, 48]`, but the STL needs coordinates.
The standard `argmax` is not differentiable (zero gradient everywhere). The
**soft-argmax** (Sun et al., "Integral Human Pose Regression", ECCV 2018)
computes the expected position instead:

```
p(i,j) = softmax(β · h(i,j))        ← normalize to probability distribution
x̂ = Σ_j  j · Σ_i p(i,j)            ← expected column
ŷ = Σ_i  i · Σ_j p(i,j)            ← expected row
```

**Temperature β.** With our Gaussian heatmaps (σ=2, peak ≈ 1.0):
- β = 1: softmax too flat → coordinates collapse to center → useless
- β = 100: softmax too sharp → near-delta → vanishing gradients
- β ≈ 10: the original choice — precise (sub-pixel) coordinates with healthy gradients

We initially set **β = 10**. A later empirical analysis of the train/eval
decoder gap (Section 11.9) showed β = 10 produces a large mismatch between the
soft-argmax coordinates the STL optimizes and the argmax coordinates the AVR
measures — one of the causes of our first failed run. We therefore raised it to
**β = 50** for fine-tuning. This is now centralized as `STL_BETA` in `config.py`.

### 11.2 Term 1: Bone Ratio Loss

Two sub-terms with **different penalty shapes**, each justified by physics:

**(a) Inter-segment ratios** (3 rules × 2 sides = 6 checks). For each
anatomical ratio (e.g. forearm/upper-arm), compute from predicted coordinates
and apply a **log-cosh penalty** centered on Winter's nominal in log space:

```
ratio  = ‖elbow - wrist‖ / ‖shoulder - elbow‖
z      = (log(ratio) - log(nominal)) / BONE_SCALE
L      = log-cosh(z)   where log-cosh(x) ≈ x²/2 near 0, ≈ |x| for large x
```

**Why log-cosh instead of hinge?** Inter-segment ratios compare *different*
bones. Monocular 2D projection means each bone can be foreshortened
independently: a bone pointing toward the camera at 75° projects to
cos(75°) ≈ 0.26 of its true length. A Winter ratio of 0.785 can legitimately
appear anywhere from ~0.20 to ~3.0 in 2D. A hard hinge at [0.55, 1.05]
produces loss ~37 on valid foreshortened poses and dominates the gradient with
a few outliers. Log-cosh is robust: it tolerates projection variance, then
grows smoothly beyond it.

**Why log space?** Makes the penalty symmetric: ratio=2.0 and ratio=0.5
receive equal penalization (a ratio and its reciprocal are equally "wrong").

**BONE_SCALE = 1.35** — the log-distance from a nominal to the boundary of
geometrically expected foreshortening: `log(1/cos(75°)) = log(1/0.259) ≈ 1.35`.
One geometric parameter, not tuned on the dataset.

Nominals from Winter 2009 / Drillis & Contini 1966:

| Ratio | Nominal |
|-------|---------|
| Forearm / Upper arm | 0.785 |
| Shank / Thigh | 1.004 |
| Upper arm / Thigh | 0.759 |

**(b) Left/right symmetry** (4 checks). Same bone on left vs right side,
penalized via **hinge on |log(ratio)|** in log space:

```
L = relu(|log(left_len / right_len)| - log(1.5))²
```

**Why hinge (not log-cosh) here?** Symmetry compares the *same* bone on both
sides. Foreshortening is approximately shared (the two limbs are roughly
coplanar), so deviation from ratio=1 signals a prediction error, not
projection geometry. The threshold `log(1.5)` fires exactly when
`max/min > 1.5` — identical to the `bone_ratio` category of the AVR metric.
Log space makes it symmetric: ratio=2.0 and ratio=0.5 are penalized equally.

### 11.3 Term 2: Joint Angle Loss

**4 joints checked** (left/right elbow, left/right knee — shoulders and hips
removed because their ROM is nearly 180° and the penalty was almost always
zero). The angle at each joint vertex is computed with **atan2**, not arccos:

```python
cross = v1.x * v2.y - v1.y * v2.x    # 2D cross product (scalar)
dot   = v1 · v2                        # dot product
angle = atan2(|cross|, dot)            # stable in [0, π]
```

**Why atan2 instead of arccos?** arccos has derivative −1/√(1−cos²θ), which
explodes at θ = 0 and θ = π — exactly the boundaries of our physiological
range. atan2 has stable derivatives everywhere.

Floor is `AVR_ANGLE_MIN_DEG = 20°` (from `config.py`, shared with AVR metric).
The upper bound 180° is inert with atan2 (angle ∈ [0, π]) but kept for
documentation symmetry with the AVR. Floor is generous to absorb 2D
foreshortening (clinical minimum is 30–40°).

### 11.4 Term 3: Bone Collapse Loss

4 kinematic chains (left/right arm, left/right leg). Each chain (proximal →
joint → distal) is checked for collapse: each sub-segment must be at least
`COLLAPSE_THRESHOLD` (10%) of the torso length.

```
torso  = ‖shoulder_mid - hip_mid‖
d1     = ‖proximal - joint‖ / torso
d2     = ‖joint - distal‖   / torso
L      = relu(COLLAPSE_THRESHOLD - d1)² + relu(COLLAPSE_THRESHOLD - d2)²
```

This directly mirrors the `collapse` category of the AVR metric: same 4
chains, same torso scale, same threshold (`COLLAPSE_THRESHOLD = 0.10` in
`config.py`). A joint that collapses onto one of its neighbors — geometrically
valid for angles but anatomically meaningless — is thus penalized during
training with the same criterion used to measure it at evaluation.

### 11.5 Term 4: Geometric Ordering Loss

4 kinematic chains (left/right arm, left/right leg). For each chain
(a → mid → b), the intermediate joint must project between the extremes:

```
t = dot(mid − a, b − a) / ‖b − a‖²
L = max(0, −t)² + max(0, t − 1)²
```

`t ∈ [0,1]` means mid is between a and b → zero penalty.
`t < 0` or `t > 1` → mid is outside the chain → quadratic penalty.

### 11.6 Combined Loss

```
L_total = L_heatmap + λ_bone     · L_bone
                    + λ_angle    · L_angle
                    + λ_order    · L_order
                    + λ_collapse · L_collapse
```

The `SkeletalTopologyLoss` class in `stl.py` wraps all terms and returns both
the total loss and a per-term breakdown dict for logging. All terms receive a
`valid_mask` derived from `target_weight` to skip unannotated keypoints (their
heatmaps are zero-target noise whose coordinates would corrupt the loss).

STL fine-tuning hyperparameters are centralized in `config.py`:
`STL_FINE_TUNE_LR`, `STL_TARGET_FRAC`, `STL_NUM_EPOCHS`, `STL_BETA`.
AVR/STL shared thresholds: `BONE_RATIO_THRESHOLD`, `COLLAPSE_THRESHOLD`,
`MIN_CONF`, `AVR_ANGLE_MIN_DEG`.

### 11.7 Design Choices Summary

| Choice | Why |
|--------|-----|
| Soft-argmax β=10 → β=50 | β=10 was the initial choice for σ=2 heatmaps; raised to β=50 after the decoder-gap analysis (Sec 11.9) showed β=10 mismatches the eval-time argmax decoder |
| atan2 not arccos | Stable gradients at range boundaries (0° and 180°) |
| Log-cosh for inter-segment ratios | Robust to 2D foreshortening (outlier tolerance) while staying smooth at center |
| Hinge² for symmetry + angles + ordering | Zero inside range, quadratic outside; physical: symmetric bones have correlated foreshortening |
| Log space for ratio penalties | Symmetric: ratio=2.0 and ratio=0.5 penalized equally |
| BONE_SCALE = 1.35 | Geometric: `log(1/cos(75°))` — tolerance boundary from expected monocular foreshortening |
| valid_mask on all STL terms | Unannotated keypoints have garbage coordinates; masking prevents loss explosion |
| Thresholds in config.py | STL and AVR share the same constants — a drop in one is guaranteed to reflect the other |
| Absolute not statistical | Key differentiator from Han et al. (2025) |

### 11.8 Differentiability Verification

`test_stl.py` runs `torch.autograd.gradcheck` on every component:

```
$ python test_stl.py

soft_argmax                    OK
bone_ratio_loss                OK
joint_angle_loss               OK
collapse_loss                  OK
geometric_ordering             OK
combined_e2e                   OK

All tests passed. STL is differentiable and ready for training.
```

### 11.9 First STL Run: Failure, Diagnosis, and Fixes

We document this because the first STL fine-tuning run **failed**, and the
diagnosis shaped the final design. Configuration: LR = 1e-4, value-based λ
auto-calibration, β = 10, 10 epochs.

**What happened.** Both objectives got worse. AP collapsed from 0.498 to 0.259.
And — more damning — the *best* epoch already had AVR = 0.46, **worse** than the
baseline's 0.306. A loss designed to reduce anatomical violations was producing
more of them. That rules out a simple "lower the weights" fix: something was
structurally wrong.

| Epoch | AP | AVR rate |
|-------|------|----------|
| baseline | 0.498 | 0.306 |
| E01 (best AP) | 0.484 | 0.464 |
| E04 | 0.337 | 0.944 |
| E10 | 0.259 | 0.824 |

**Three root causes:**

1. **Inverted λ auto-calibration.** The original heuristic set each weight as
   `λ_k = TARGET_FRAC · L_hm / (raw value of term k)`. But a term's raw value is
   *small precisely when the baseline already satisfies it*. The angle term sits
   near zero on the baseline (angles already in range), so dividing by it
   produced a huge `λ_angle ≈ 1.5`, ~300× the bone weight. The moment a few poses
   left the angle range during training, they received an enormous gradient; the
   model chased angles and abandoned the heatmaps. The heuristic rewarded the
   already-satisfied term with the largest weight — the exact opposite of what we
   want.

2. **Learning rate 10× too high.** The intended fine-tuning LR was ~2e-5; the
   code ran 1e-4. On already-converged weights this is textbook catastrophic
   forgetting (the 1.5-point AP drop by E01, the cliff at E04).

3. **Train/eval decoder mismatch.** The STL optimizes soft-argmax coordinates;
   the AVR measures argmax + sub-pixel coordinates. They are two different
   read-outs of the same heatmap. The STL can make *its* coordinates more
   plausible while the *measured* coordinates degrade — one way a
   violation-reducing loss raises the measured violation rate.

The masking and the four terms were correct all along. The failure was in the
training **harness** — weight calibration, learning rate, and an unverified
assumption that two decoders agree.

### 11.9.1 Decoder gap analysis → β = 50

To quantify cause (3), we measured the per-keypoint Euclidean gap (in heatmap
pixels) between `soft_argmax(β)` and `decode_heatmaps` (argmax + sub-pixel) on
COCO val, on valid keypoints only:

| β | median gap | p95 | max |
|------|-----------|------|------|
| 10 | 4.12 px | 21.1 | — |
| 30 | 0.26 px | 9.3 | 32 |
| 50 | 0.27 px | 3.2 | — |
| 100 | 0.31 px | 0.79 | — |

The median is reassuring at β ≥ 30, but the **heavy tail** (p95, max) is the
real story: it comes from multimodal/flat heatmaps on occluded keypoints
(wrists, hips, ankles) — exactly the joints the STL must act on. When a heatmap
has two peaks, argmax picks one while soft-argmax averages to the middle, where
no peak exists. Raising β tames the tail (p95: 9.3 → 0.79 from β=30 to β=100)
without changing the median, but very high β flattens the gradient. We chose
**β = 50** as the compromise (p95 = 3.2 px, gradients still usable) and document
the sweep as an ablation. This also explains why the first run (β = 10) failed
even setting the λ issue aside: a 4 px median gap means the STL was optimizing
coordinates almost unrelated to those measured.

### 11.10 λ Calibration: Gradient-Norm with Spread Clamp

The replacement for the broken value-based heuristic. The insight: a λ does not
control "how much we care" about a term — it controls **how strongly that term's
gradient pushes the weights** at each step. Different terms have wildly different
gradient scales, so equal λ does not mean equal influence.

**Gradient-norm calibration (GradNorm-style, static; Chen et al. 2018).** For
each unweighted term we measure the gradient norm w.r.t. the shared final layer
(the 1×1 Conv producing the 17 heatmaps — the bottleneck where all gradients
live on the same weights and are comparable), and set:

```
λ_t = ρ · g_hm / (g_t + ε)
```

so each constraint imprints a fraction ρ (`STL_TARGET_FRAC = 0.1`) of the
heatmap loss's push. This is the **opposite** of the old bug: a term that
already pushes hard (large g_t) gets a small λ; a quiet term gets a larger one —
influence equalized, not value. Computed once on the baseline before
fine-tuning (we keep `mode='per_epoch'` / full GradNorm for future work).
Implemented in `stl.py::calibrate_lambdas`; gradient isolation per term is
verified (zeroing grads between terms, no accumulation).

**The instability problem and the spread clamp.** Run on the baseline, the raw
calibration produces a pathological spread, and — crucially — an **unstable**
one. Two independent runs of the calibration gave:

| run | bone | angle | order | collapse | max/min spread |
|-----|------|-------|-------|----------|----------------|
| A | 8.9e-5 | 1.2e-3 | 3.9e-5 | 4.9e-2 | **1236×** |
| B | 1.4e-4 | 3.3e-3 | 1.2e-5 | 1.9e-1 | **15573×** |

The spread is enormous *and* jumps by an order of magnitude between identical
runs. The reason: terms like `collapse` and `order` are already nearly satisfied
by the baseline, so their gradient norm is dominated by a handful of violating
poses that happen to fall in the 4 sampled batches. With so few samples, the
norm of an almost-satisfied term is essentially noise — its value (and the
resulting λ) is not reproducible. An un-clamped calibration would hand a
near-satisfied term a 10⁴× weight, recreating exactly the failure mode that sank
the first run.

We therefore clamp the spread. After computing the raw λ, we compress them
around their geometric mean so that `max/min ≤ max_spread` (default 20×), in log
space for multiplicative symmetry:

```python
λ_t = exp( clamp( log(λ_t) − center, ±½·log(max_spread) ) + center )
```

With `max_spread = 20` the example above becomes
`bone=1.5e-4, angle=1.2e-3, order=1.5e-4, collapse=3.0e-3` — no term dominates.
Note `bone` and `order` end up equal: their raw difference (≈2×) is well within
the sampling noise of a near-satisfied term measured on 4 batches, so the clamp
correctly declines to distinguish quantities it has no reliable signal to
separate. The instability across runs (1236× vs 15573×) is itself the
experimental justification for the clamp: it is not cosmetic, it is what makes
the calibration reproducible.

This is also a free diagnosis for the paper: terms with near-zero baseline
gradient norm (collapse) are constraints the baseline already respects; terms
with large norm (order, bone) are where violations exist and the STL will act.

---

## 12. Still To Do

### 12.1 STL Training run

The harness is now fixed (Section 11.9–11.10): β = 50, LR = `STL_FINE_TUNE_LR`
(3e-5), λ from gradient-norm calibration with spread clamp, selection on AP (not
val_loss), all four terms including collapse. What remains is to **run** the
fine-tuning from `best.pth` and read the result. Protocol: start with 1 epoch as
a sanity check (AP must hold within ~1 point of 0.498, AVR must drop below
0.306), then run the full schedule. Per-epoch checkpoints are saved so the
AP/AVR trade-off can be inspected by hand.

### 12.2 Grad-CAM Explainability

Grad-CAM on the last convolutional layer of the backbone, targeting specific
keypoint channels. Compare baseline vs STL on occluded examples: does the STL
model attend to neighboring visible joints?

### 12.3 Ablation Study

Term ablation (the STL has **four** terms now):
1. Heatmap loss only (baseline) — done
2. Heatmap + bone ratio
3. Heatmap + bone ratio + joint angle
4. Heatmap + bone ratio + joint angle + collapse
5. Full STL (+ geometric ordering)

Already available as ablations from this work:
- **β sweep** (decoder gap vs β, Section 11.9.1) — justifies β = 50
- **λ spread clamp** (Section 11.10) — un-clamped vs clamped, stability across runs
- **λ sensitivity** (trade-off AP vs AVR at different `STL_TARGET_FRAC`)

### 12.4 Inference Benchmark

Measure ms/image on CPU and GPU. Compare with ViTPose and HRNet. Count FLOPs
with `thop` or `fvcore`.

### 12.5 Paper and Presentation

ICRA/IROS format. 15-20 slides for the exam. Failure case analysis with
side-by-side visualizations.

---

## 13. How to Run

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

## 14. Reproducibility

- Seed: `SEED = 42`, applied to `random`, `numpy`, `torch`, `torch.cuda`,
  `cudnn.deterministic = True`, `cudnn.benchmark = False`
- Framework: Python 3.10, PyTorch (Kaggle default), torchvision
- GPU: NVIDIA T4 (Kaggle) for the baseline; local RTX 5070Ti for STL
  fine-tuning after the Kaggle GPU quota was exhausted. Note that
  `num_workers > 0` introduces minor non-determinism in batch ordering. The λ
  calibration is also sampling-sensitive on few batches (Section 11.10) — hence
  the spread clamp.
- All hyperparameters in `config.py`
- Training history logged to `history.csv`

---

## 15. References

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
7. **Integral Pose** — Sun et al., "Integral Human Pose Regression",
   ECCV 2018 (soft-argmax / integral regression)

### Biomechanical sources
- Winter, D.A. "Biomechanics and Motor Control of Human Movement", 4th ed., Wiley, 2009
- Drillis, R. & Contini, R. "Body Segment Parameters", Technical Report No. 1166-03, NYU, 1966
- American Academy of Orthopaedic Surgeons (AAOS), "Joint Motion: Method of Measuring and Recording", 1965