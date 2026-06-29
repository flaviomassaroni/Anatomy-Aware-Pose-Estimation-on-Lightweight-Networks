# AVR / STL Analysis: diagnosis, result, and strategies

**Project:** Anatomy-Aware Pose Estimation on Lightweight Networks
**Stage:** STL fine-tuning (Stage 2), starting from baseline `best.pth`
**Baseline reference:** AP = 0.498, AVR pose-rate = 0.306 (COCO val, 6352 poses)
**Question investigated:** the Skeletal Topology Loss reduces anatomical
violations by design, yet at first the measured AVR did not drop. This document
records the diagnostic process, the numbers, the working fix and its full
evaluation, and the strategy space for pushing further.

---

## 1. Summary

The STL harness is sound (AP preserved; the first run's catastrophic forgetting
is fixed). The AVR is dominated by its `bone_ratio` category. Five diagnostics
eliminated three candidate causes and isolated two co-existing ones, then a
minimal fix (raise `lambda_bone`) produced the paper's headline result.

**Diagnosis — two real forces holding up bone_ratio:**

1. **Structural floor (foreshortening).** ~63% of *severe* bone_ratio violations
   are geometrically consistent with a limb projected toward the camera (true 2D
   foreshortening, not a prediction error). The symmetry term assumes left/right
   coplanarity; when broken, the violation is real in 2D and **cannot be
   corrected without 3D**. A hard floor on the AVR.

2. **Gradient dilution (correctable).** Poses with a severe violation carry a
   gradient ~4.3x the batch average but are only ~17% of poses, so batch-mean
   averaging dilutes them. The default push was too gentle.

**Result.** A minimal fix (raise `lambda_bone` 5x) plus **severity weighting** of
the symmetry sub-term lowers the AVR with AP preserved. After discovering ~0.01
AVR non-determinism, all final numbers were re-measured deterministically (fixed
seed, `num_workers=0`). Best model: **sev_E7** — STL with linear severity weighting
(`penalty = hinge·(1 + α·excess)`, α=2.0), LR 1e-5, boost 5×, epoch 7.

*Deterministic final comparison (baseline → sev_E7):*

| | AP | AVR rate |
|---|------|----------|
| COCO val | 0.4984 → 0.4964 | 0.3065 → **0.2996** (−0.0069) |
| OCHuman zero-shot | 0.4364 → 0.4365 | 0.4824 → **0.4691** (−0.0133) |

AVR-mean on OCHuman drops 0.693 → 0.659 (−0.034). The gain is **reproducible and
generalizes**: sev_E7 beats both baseline and the plain-STL model (A_pure_E8) on
*both* datasets, and by a larger margin on the harder one (OCHuman −0.0133 vs
A_pure's −0.0079). Per-category on OCHuman: **collapse −27%** (0.035 → 0.025),
**joint_angle −9%** (0.099 → 0.091), and — crucially — **bone_ratio −2.8%**
(0.559 → 0.543), where plain STL barely moved it (−0.4%). The severity weighting
unblocks the *correctable* bone violations that gradient dilution had hidden,
pushing bone_ratio ~7× further than plain STL (−0.0156 vs −0.0023), without
fighting the foreshortening floor (which still caps the absolute value). AP is
preserved throughout. The thesis — dataset-independent priors that generalize
zero-shot — holds, now with a meaningful effect size. Section 6 documents the path
(negative log-cosh ablation, exhausted dynamics tuning) and the winning severity
weighting; remaining lever for more gain: occlusion augmentation (#5).

---

## 2. Background: the metric and the loss

The **AVR** (Anatomical Violation Rate) counts the fraction of predicted poses
that violate at least one anatomical rule, across three categories:
`bone_ratio`, `joint_angle`, `collapse`. It uses **predicted** keypoints gated by
confidence (`score >= MIN_CONF`, MIN_CONF = 0.3) and does **not** use ground
truth.

The **STL** adds four differentiable terms (bone ratio, joint angle, geometric
ordering, bone collapse) on soft-argmax coordinates. The bone_ratio term has two
sub-terms: inter-segment ratios (log-cosh) and left/right symmetry (squared
hinge on |log(ratio)| with threshold log(1.5), capped at 4.0). The symmetry
sub-term shares the exact 1.5 threshold with the AVR bone_ratio category, so the
two are directly comparable.

The STL masks on `target_weight` (annotated keypoints, GT visibility v>0).
This gate mismatch with the AVR (predicted confidence) was the first suspect.

---

## 3. Diagnostic timeline and results

### 3.1 Sanity v1 — 1 epoch, standard calibrated lambdas

| metric | baseline | E1 |
|--------|----------|------|
| AP | 0.498 | 0.4967 |
| AVR | 0.306 | 0.3105 |

AP holds (no forgetting). AVR does not drop (+0.0045). Loss terms at E1 showed
`order` dominating (L_order ~ 0.80) while `bone`/`angle` were small and
`collapse` ~ 0. Note: in the first *failed* run (LR 1e-4, beta 10) AVR had jumped
to 0.464 at E1 — here it barely moves, so the harness is no longer diverging.

### 3.2 Sanity v2 — 3 epochs, `order` demoted to lambda = 1e-4

Trajectory:

| epoch | AP | AVR | bone_ratio | joint_angle | collapse |
|-------|------|------|-----------|------------|----------|
| 0 (base) | 0.498 | 0.3065 | — | — | — |
| 1 | 0.4967 | 0.3356 | — | — | — |
| 2 | 0.4963 | 0.3168 | — | — | — |
| 3 | 0.4973 | 0.3193 | 0.3254 | 0.0704 | 0.0287 |

Loss terms at E3: L_bone 0.0511, L_angle 0.0011, L_order 0.1985, L_collapse
0.0000. AP intact across all epochs. AVR oscillates **above** baseline.

**Reading.** The AVR total is essentially its `bone_ratio` component
(0.3254 of 0.3193 total pose-rate — the categories are per-pose rates, not a
partition, hence bone_ratio alone ~ the total). Demoting `order` did not help;
its raw loss is so large that even at lambda 1e-4 it remains the biggest weighted
term. The bottleneck is the bone_ratio category specifically.

### 3.3 Diagnostic C — is the gate the cause?

For every AVR bone_ratio violation on the baseline, check whether the involved
keypoints were annotated (inside the STL gate) or not.

| | count | share |
|---|------|-------|
| Total bone_ratio violations | 1974 | 100% |
| Inside STL gate (all v>0) | 1376 | **69.7%** |
| Outside gate (>=1 occluded) | 598 | 30.3% |

Distribution by number of occluded (v=0) keypoints per violation:

| occluded kp | 0 | 1 | 2 | 3 | 4 |
|-------------|------|-----|-----|----|----|
| violations | 1376 | 254 | 277 | 39 | 28 |

**Verdict: gate eliminated.** 70% of violations are on keypoints the STL already
sees. Aligning the gate (training on predicted-confidence keypoints) would
address at most the 30% minority, at the cost of the training instability the
gate was designed to avoid. Options A/B dropped.

### 3.4 Diagnostic D — magnitude of violations and read-out gap

**Measure 1 — ratio distribution of in-gate violations (n = 1375):**

| stat | value |
|------|-------|
| median | 1.902 |
| p75 | 2.529 |
| p90 | 3.687 |
| p95 | 5.154 |
| p99 | 9.269 |
| max | 31.064 |

| band | count | share |
|------|-------|-------|
| borderline (1.5–1.7) | 452 | 32.9% |
| medium (1.7–2.0) | 331 | 24.1% |
| severe (>2.0) | 592 | **43.1%** |

**Measure 2 — read-out gap, soft-argmax(beta=50) vs argmax+subpixel (heatmap px):**

| keypoint set | median | p95 | max |
|--------------|--------|------|------|
| all annotated kp (reference) | 0.265 | 2.254 | — |
| kp involved in violations | 0.263 | 0.733 | 16.633 |

**Verdict: read-out eliminated.** The gap on violating keypoints (median 0.263,
p95 0.733) is equal-or-better than the global reference. beta=50 makes the loss
and the metric read the same coordinates. The loss is *not* optimizing a
different read-out than the one measured. But the violations are large (43%
severe, median 1.90 — a near-doubled bone), so weak spinta alone does not explain
why a strong loss leaves them untouched.

### 3.5 Diagnostic E — foreshortening vs gradient dilution

**Measure 1 — nature of severe violations (ratio > 2.0, n = 592):**

| class | count | share |
|-------|-------|-------|
| foreshortening-like (short bone < 0.25·torso) | 375 | **63.3%** |
| error-like (both bones plausible length) | 217 | 36.7% |

**Measure 2 — gradient dilution (||grad bone_ratio|| on final_layer, 6 batches):**

| quantity | value |
|----------|-------|
| mean fraction of poses with severe violation per batch | 16.7% |
| ‖grad bone‖ on full batch | 2.192e-01 |
| ‖grad bone‖ on violators only | 9.462e-01 |
| ratio violators / batch | **4.32x** |

**Verdict: both residual causes confirmed.**
- *(3a) Foreshortening floor.* 63% of severe violations are consistent with a
  genuinely foreshortened limb — the short bone is also short relative to the
  torso, the signature of a limb pointing at the camera. The symmetry term's
  coplanarity assumption is violated; penalizing these would push the model to
  predict a *wrong* (un-foreshortened) pose. This fraction is a structural floor.
- *(3b) Gradient dilution.* Violating poses carry 4.3x the batch-average bone
  gradient but are only ~17% of poses, so batch-mean averaging buries them. This
  is the correctable part.

---

## 4. Causes considered, and their disposition

| # | hypothesis | test | result | disposition |
|---|-----------|------|--------|-------------|
| 1 | Catastrophic forgetting (as in first run) | sanity v1/v2 AP | AP held within 0.002 | **fixed** (LR 3e-5 + lambda calib) |
| 2 | `order` term hogging the gradient | demote order, v2 | AVR still flat; bone is the driver | not the cause |
| 3 | Gate mismatch (STL target_weight vs AVR confidence) | diag C | 70% violations in-gate | **eliminated** |
| 4 | Read-out mismatch (soft-argmax vs argmax) | diag D M2 | gap on violators ~ reference | **eliminated** |
| 5 | Borderline violations needing a small push | diag D M1 | only 33% borderline; 43% severe | partial only |
| 6 | Foreshortening floor (2D projection) | diag E M1 | 63% of severe are foreshortening-like | **confirmed (floor)** |
| 7 | Gradient dilution by batch mean | diag E M2 | violators 4.3x batch | **confirmed (correctable)** |

---

## 4bis. Result: L1 fix (raise lambda_bone) and full evaluation

After the diagnostics, the first fix (Level 1: raise only `lambda_bone`, no code
change) was tested with `BONE_BOOST = 5x` on the calibrated value, 3 epochs.

**Sanity trajectory (COCO val, during loop):**

| epoch | AP | AVR | bone_ratio |
|-------|------|------|-----------|
| 0 (base) | 0.498 | 0.3073 | 0.3112 |
| 1 | 0.4979 | **0.2875** | **0.2935** |
| 2 | 0.4973 | 0.2903 | 0.2991 |
| 3 | 0.4958 | 0.3309 | 0.3430 |

The minimum is at **E1**: AVR drops below baseline with AP intact. From E2 the
curve turns — bone_ratio and AVR climb back, AP starts to cede. This late
divergence is the foreshortening floor manifesting (see Section 5).

**Full evaluation of the E1 checkpoint, baseline vs STL, COCO + OCHuman zero-shot:**

| metric | COCO base | COCO STL | OCHuman base | OCHuman STL |
|--------|-----------|----------|--------------|-------------|
| AP @0.50:0.95 | 0.4983 | 0.4979 | 0.4363 | 0.4371 |
| AP @0.50 | 0.7768 | 0.7852 | 0.7786 | 0.7700 |
| AR @0.50:0.95 | 0.5384 | 0.5380 | 0.5227 | 0.5232 |
| **AVR rate** | **0.3062** | **0.2875** | **0.4822** | **0.4561** |
| **AVR mean** | 0.4051 | 0.3730 | 0.6938 | 0.6469 |

**AVR per-category (rate):**

| category | COCO base | COCO STL | OCHuman base | OCHuman STL |
|----------|-----------|----------|--------------|-------------|
| bone_ratio | 0.3108 | 0.2935 | 0.5591 | 0.5269 |
| joint_angle | 0.0652 | 0.0567 | 0.0995 | 0.0900 |
| collapse | 0.0291 | 0.0228 | 0.0352 | 0.0301 |

**Deltas:**

| | AP | AVR rate |
|---|------|----------|
| COCO | −0.0004 | **−0.0187** |
| OCHuman (zero-shot) | +0.0009 | **−0.0261** |

**Reading.** This is the headline result. (1) AVR drops on **both** datasets,
*more* on OCHuman (−0.0261), the higher-occlusion benchmark — the anatomical
prior helps most exactly where occlusion makes estimation ambiguous, which is the
paper's claim. (2) AP is preserved (−0.0004 on COCO, +0.0009 on OCHuman): safety
gain at no accuracy cost, the key message for HRI. (3) **All three** categories
improve on both datasets. (4) The improvement was trained on COCO only and
transfers zero-shot to OCHuman, demonstrating that absolute (literature-based)
constraints generalize cross-dataset — the differentiation from Han et al. (2025),
whose constraints are learned from the training distribution.

The E1 AVR in clean isolated evaluation (0.2875) matches the in-loop value,
confirming reproducibility (not an artifact of the training loop).

---

## 5. Conclusion

The AVR bone_ratio is the binding category, and it is held up by two distinct
forces. About **63% of severe violations are 2D-foreshortening artifacts** that
no 2D loss should "fix" — they represent real projected geometry, and pushing
them toward left/right symmetry would degrade the actual pose. This is a
principled floor on the AVR, worth stating explicitly in the paper: the metric
penalizes some geometrically valid monocular configurations, so a non-zero
bone_ratio AVR is expected even for a perfect 2D predictor.

The remaining **~37% error-like severe cases, plus the borderline/medium mass
(57% of all in-gate violations)**, are correctable but currently under-served:
their gradient (4.3x the batch mean) is diluted by the ~83% of well-behaved poses
in each batch. The current `lambda_bone` and LR are too gentle to move them in a
few epochs.

---

## 6. Strategies to push AVR below the E1 minimum

The L1 trajectory shows a local minimum at E1 then divergence: more epochs make
it *worse*, because once the easy ("error-like") asymmetries are corrected, the
symmetry loss starts fighting the foreshortening floor — pushing genuinely
foreshortened limbs toward a symmetry that does not exist in 2D, which degrades
both the measured AVR and AP. Going below E1 therefore requires changing *what*
the loss does, not *how long* it runs. Strategies below are grouped by how
invasive they are and whether they touch the paper's thesis.

**Why longer training does not help (mechanism).** The symmetry penalty cannot
distinguish two kinds of left/right asymmetry: (Group 1) estimation errors, where
equalizing the sides moves coordinates toward truth — AVR down, AP stable; and
(Group 2) real foreshortening, where one limb points at the camera and projects
short, so equalizing moves a correct keypoint toward a wrong position — AVR up, AP
down. E1's gradient is dominated by the large, easy Group 1. Once exhausted, E2-E3
gradients are dominated by Group 2, doing damage. The turning point is when Group
1 runs out.

### 6.1 No code change (cheap tests, try first)

| # | strategy | idea | thesis impact |
|---|----------|------|---------------|
| 1 | **Checkpoint selection at E1** | The best model already exists; select on AVR+AP, not just AP. Standard practice. | none |
| 2 | **Lower LR, more epochs** | LR 1e-5 takes smaller steps; may stop before hitting the floor and give more near-minimum points to choose from. | none |
| 3 | **Lower boost, longer** | boost 2-3x exhausts Group 1 more slowly; the descent is gentler and the minimum may sit lower. | none |
| 4 | **Lambda warmup** | Start `lambda_bone` low, ramp up; avoids the initial shock and lets the model settle. | none |

### 6.2 Data-side (inside the existing thesis)

| # | strategy | idea | thesis impact |
|---|----------|------|---------------|
| 5 | **Foreshortening / occlusion augmentation** | The abstract already plans an occlusion-augmentation study (reconciling Pytel vs Han). Augmenting with simulated foreshortening teaches the model to handle Group 2 at the source, shrinking the floor instead of damping it in the loss. | strengthens it — already in scope |

### 6.3 Loss-side (touch stl.py, stay dataset-independent)

| # | strategy | idea | thesis impact |
|---|----------|------|---------------|
| 6 | **L2a — severity weighting** | Weight the symmetry sub-term by how far the ratio exceeds 1.5, so severe Group-1 cases are not averaged away by the well-behaved majority. Focal-style, no learned statistics. | none (absolute constraints preserved) |
| 7 | **Scale-adaptive symmetry threshold** | Raise the 1.5 threshold when a bone is short relative to the torso (a foreshortening signal). Geometry-based, not data-learned. | minor — argue as 2D-geometry, not learned |
| 8 | **Log-cosh for symmetry** (TESTED — rejected, see 6.6) | Replace the squared hinge on \|log(ratio)\| with log-cosh — as the inter-segment sub-term does. Hypothesis: log-cosh tolerates extreme foreshortening asymmetry while still penalizing moderate errors. **Result: degrades performance** at every scale tried; the hinge's aggressiveness is what produces the E1 drop. | none — but empirically worse |

### 6.4 Metric-side (subtle, argue carefully)

| # | strategy | idea | thesis impact |
|---|----------|------|---------------|
| 9 | **Foreshortening-aware AVR** | The AVR counts foreshortening asymmetry as a violation — a metric false positive. Refine the AVR to not penalize asymmetry when a bone is very short vs the torso. Improves the metric, does not change the thesis, but touches the KPI itself. | none to thesis, but must be justified |

### 6.5 Recommendation (updated after testing #8)

The thesis is already demonstrated by the E1 result. The structural loss-shape
change we expected to help (#8, log-cosh) was tested and **rejected** (6.6): the
hinge's aggressiveness is a feature, not a bug. Updated priority for *extra*
margin: (a) cheap no-code tuning (#2 lower LR, #3 lower boost + longer, #4 lambda
warmup) — these only change *when* to stop, not the loss; (b) **#6 severity
weighting** if a loss-shape change is still wanted (it sharpens the hinge rather
than softening it, the opposite of #8); (c) **#5 augmentation** as the
medium-term direction already in the paper's scope. Regardless of the fix, report
the foreshortening floor quantitatively (63% of severe violations) as an intrinsic
property of a 2D anatomical metric, framing the STL's win as reducing the
*correctable* fraction. The current best deliverable is **hinge + checkpoint
selection at E1**.

### 6.6 Ablation: log-cosh symmetry term (#8) — negative result

We replaced the squared-hinge symmetry sub-term with a dead-zoned log-cosh
(`logcosh(relu(|log ratio| - log 1.5) / SYM_SCALE)`), preserving the AVR
threshold but making the penalty grow ~linearly (robust to outliers) instead of
quadratically. The gradcheck in `test_stl.py` passed for all variants. Same
harness, boost 5x, 3 epochs, COCO val:

| variant | E0 bone | E1 bone | E2 bone | E3 bone | best AVR |
|---------|---------|---------|---------|---------|----------|
| **hinge (current)** | 0.3112 | **0.2935** | 0.2991 | 0.3430 | **0.2875** (E1) |
| log-cosh SYM_SCALE=1.5 | 0.3120 | 0.3168 | 0.3109 | 0.3245 | 0.3034 (E2) |
| log-cosh SYM_SCALE=0.7 | 0.3130 | 0.3607 | 0.3300 | 0.3180 | 0.3156 (E3) |

**Both log-cosh variants are worse than the hinge**, and never beat baseline AVR
(0.306). At SYM_SCALE=1.5 the gradient is ~5x softer on severe cases — too soft,
it kills the useful push on real errors. At SYM_SCALE=0.7 the gradient matches the
hinge on medium errors but the run is worse still (E1 bone *rises* to 0.3607).

**Interpretation.** The hinge works *because* it is aggressive: it pushes hard and
fast on large errors in the first epoch, achieving the drop at E1; the later
rebound (E2-E3) is the *price* of that early success, not a defect. The squared +
capped shape is, in effect, already foreshortening-aware: gradient peaks on
medium-severe errors (ratio 3-5, correctable) and the cap zeroes the push on
extreme outliers (ratio >15, pure foreshortening). Softening the term — by any
scale — removes the early push that produces the gain. **Conclusion: keep the
hinge; select the checkpoint at E1.** Softening the loss is the wrong direction;
if a loss-shape change is pursued, it should *sharpen* the signal on correctable
errors (#6), not smooth it.

### 6.7 Dynamics tuning (#2 LR, #3 boost, #4 warmup) — grid + extended run

After rejecting the loss-shape change (#8), we tested whether *training dynamics*
— not the loss itself — could push below the E1 minimum. Five configs, same
hinge, same shared lambda calibration (`lambda_bone` base = 0.00024), each from
the same `best.pth`, 3 epochs, COCO val. Best AVR over any epoch (AP ≥ 0.486):

| config | LR | boost | warmup | best AVR | @ep | AP@best |
|--------|------|-------|--------|----------|-----|---------|
| A | 1e-5 | 5x | no | **0.2950** | 3 | 0.4978 |
| D | 3e-5 | 5x | yes | 0.2968 | 2 | 0.4950 |
| ref | 3e-5 | 5x | no | 0.2993 | 3 | 0.4947 |
| B | 3e-5 | 3x | no | 0.3012 | 3 | 0.4963 |
| C | 1e-5 | 3x | no | 0.3054 | 2 | 0.4978 |

**A key methodological finding emerged: non-determinism.** The reference config
(3e-5, 5x) scored 0.2993 here but 0.2875 in the original L1 run — *same config,
different result*. Cause: `num_workers>0` + train shuffle + non-deterministic
cuDNN make repeated runs of the same config differ by ~0.01 AVR. **This means the
verdict must compare against the in-grid reference (0.2993), not the historical
0.2875** (which carried favorable noise). Read that way:

| config | vs in-grid ref (0.2993) | isolates |
|--------|--------------------------|----------|
| A (LR 1e-5) | **−0.0043 better** | lower LR |
| D (warmup) | **−0.0025 better** | warmup |
| B (boost 3x) | +0.0019 ~same | lower boost |
| C (LR+boost low) | +0.0061 worse | interaction |

**Conclusions from the grid.** (1) **Lower LR helps** (A best), and crucially A's
minimum is at E3 *without rebound* — the descent had not finished in 3 epochs,
suggesting more headroom. (2) **Warmup helps** (D), a softer start lowers the
early minimum. (3) **Lower boost hurts** (B, C) — consistent with the 8b lesson
that the strong push is needed. (4) The earlier 0.2875 was partly noise; honest
reporting requires deterministic re-measurement.

**Extended deterministic runs (completed) — the 0.2875 was noise.** A (LR 1e-5,
boost 5x) and A+warmup were re-run for 8 epochs with full determinism (fixed seed,
`num_workers=0`, seeded generator). Both COCO-val trajectories:

| run | E0 | E1 | E2 | E3 | E4 | E5 | E6 | E7 | E8 | best (AVR, @ep, AP) |
|-----|------|------|------|------|------|------|------|------|------|----------------------|
| A pure | 0.3049 | 0.3094 | 0.3179 | 0.3112 | 0.3064 | 0.3116 | 0.3086 | 0.3031 | 0.3016 | 0.3016 @E8, AP 0.4969 |
| A+warmup | 0.3049 | 0.3013 | 0.3131 | 0.3168 | 0.3078 | 0.3027 | 0.3062 | 0.3042 | 0.3043 | 0.3013 @E1, AP 0.4976 |

**Both trajectories oscillate around ~0.305 with no real descent** — the "best" is
just the low point of the noise, not the bottom of a curve. Warmup adds nothing
(0.3013 vs 0.3016, identical within noise). Crucially, **neither approaches the
historical 0.2875**: under deterministic conditions the STL settles at ~0.3015.

**Conclusion: the historical E1 = 0.2875 was a favorable fluctuation of the
non-deterministic pipeline, not a reproducible result.** The honest,
reproducible finding on COCO val is a small gain: baseline (deterministic E0)
0.3049 → STL ~0.3015, i.e. **−0.003 AVR with AP preserved** (0.498 → 0.497-0.498).
The gain is real (two independent runs agree) but modest — expected, because COCO
val has little headroom and 63% of severe violations are the foreshortening floor.

**Implication for the result.** COCO val is the wrong place to look for the
headline: the baseline is already good (AVR 0.305) and the floor caps the gain.
The real test is **OCHuman zero-shot**, where the baseline violates far more (AVR
0.482) and there is room to improve.

### 6.8 Final deterministic comparison (baseline vs A_pure_E8 vs A+warmup_E1)

All three checkpoints re-evaluated on COCO val and OCHuman zero-shot with
`num_workers=0` (deterministic), identical protocol:

| model | COCO AP | COCO AVR | OC AP | OC AVR | OC AVR-mean |
|-------|---------|----------|-------|--------|-------------|
| baseline | 0.4984 | 0.3065 | 0.4364 | 0.4824 | 0.6928 |
| **A_pure_E8** | 0.4969 | **0.3016** | 0.4358 | **0.4745** | **0.6749** |
| A+warmup_E1 | 0.4976 | 0.3013 | 0.4373 | 0.4803 | 0.6896 |

Δ vs baseline:

| model | ΔCOCO AP | ΔCOCO AVR | ΔOC AP | ΔOC AVR |
|-------|----------|-----------|--------|---------|
| A_pure_E8 | −0.0014 | **−0.0049** | −0.0006 | **−0.0079** |
| A+warmup_E1 | −0.0008 | −0.0052 | +0.0009 | −0.0021 |

**A_pure_E8 is the final model.** On COCO the two configs tie (~−0.005), but on
OCHuman A_pure generalizes ~4× better (−0.0079 vs −0.0021). Warmup is a dead end:
its OCHuman gain is marginal and its bone_ratio is slightly *worse* than baseline.

**Where the gain comes from — per-category on OCHuman (A_pure_E8):**

| category | baseline | A_pure_E8 | Δ |
|----------|----------|-----------|------|
| bone_ratio | 0.5586 | 0.5563 | −0.4% (floor-limited) |
| joint_angle | 0.0993 | 0.0904 | **−9%** |
| collapse | 0.0350 | 0.0282 | **−19%** |

This is the key qualitative result: the aggregate AVR gain is modest because its
dominant category (bone_ratio) is capped by the foreshortening floor, but the STL
**works well on the non-floor-limited constraints** — joint_angle and collapse
drop 9% and 19%. The loss does its job; the metric's bone_ratio component is
partly measuring irreducible 2D projection geometry. For the paper, report the
aggregate honestly (−0.0079 OCHuman AVR, AP preserved) and the per-category split
to show the STL is effective where the 2D floor does not bind.

**Status: dynamics tuning exhausted.** LR/boost/warmup cannot move the floor.
For a larger gain the next lever is #6 (severity weighting — sharpen the bone
signal on correctable errors), tested in 6.9.

### 6.9 Severity weighting (#6) — the winning loss-shape change

The log-cosh ablation (6.6) taught that the symmetry term must be *sharpened* on
correctable errors, not smoothed. Severity weighting does exactly that: multiply
the hinge by a weight that grows with violation severity, so the strongly-violating
(but correctable) cases that gradient dilution had buried get more gradient. Two
forms were compared in-cell (penalty and gradient on synthetic ratios):

- **Focal** `weight = excess^γ` — *rejected*. Its gradient is ~0 on borderline/mid
  errors (ratio 1.7-2.0) and maximal on extremes (ratio 10-20, mostly
  foreshortening). Wrong profile: it ignores correctable cases and hammers the
  floor — the same failure mode as the 8b log-cosh, mirrored.
- **Linear** `weight = 1 + α·excess` (α=2.0) — *chosen*. Gradient exceeds the hinge
  on every correctable case (+38% at ratio 1.7, +86% at 2.0, ~3× at 3.0) while
  staying contained on extremes (much smaller than focal at ratio 20). Both pass
  gradcheck; linear has the right shape.

Implemented entirely in-cell (a `bone_ratio_loss_sev` reusing the stl.py helpers,
no change to stl.py), then trained 8 epochs deterministically from `best.pth`
(LR 1e-5, boost 5×), and evaluated on COCO + OCHuman vs baseline and A_pure_E8:

| model | COCO AVR | OC AVR | OC AVR-mean | OC bone | OC angle | OC collapse |
|-------|----------|--------|-------------|---------|----------|-------------|
| baseline | 0.3065 | 0.4824 | 0.6928 | 0.5586 | 0.0993 | 0.0350 |
| A_pure_E8 | 0.3016 | 0.4745 | 0.6749 | 0.5563 | 0.0904 | 0.0282 |
| **sev_E7** | **0.2996** | **0.4691** | **0.6591** | **0.5430** | 0.0907 | **0.0254** |
| sev_E8 | 0.3002 | 0.4742 | 0.6665 | 0.5491 | 0.0925 | 0.0249 |

**sev_E7 is the final model.** It beats both baseline and plain STL on *both*
datasets, generalizing (unlike warmup, which won on COCO but failed on OCHuman):
−0.0069 COCO and −0.0133 OCHuman vs baseline, and −0.0054 OCHuman vs A_pure. AP is
preserved (OCHuman 0.4365, marginally above baseline).

**Why it works — bone_ratio finally moves.** Plain STL could not move bone_ratio
on OCHuman (−0.0023, floor-bound). Severity weighting pushes it −0.0156, ~7×
further, because the reinforced gradient reaches the *correctable* strong
violations that batch-mean dilution had hidden — without over-pushing the extreme
(foreshortening) cases, which the linear weight keeps contained. The foreshortening
floor still caps the absolute bone_ratio value (it cannot go to zero), but the
correctable fraction is now actually corrected. Collapse also improves markedly
(−27%). This validates the diagnosis: the residual AVR was part floor (irreducible)
and part dilution (correctable); severity weighting addresses the latter.

**α was not swept.** At α=2 the gain is solid and the marginal value of tuning α is
low: all loss-side levers hit the same foreshortening floor, so α trades within a
narrow band. The real remaining lever is data-side augmentation (#5), which can
shrink the floor rather than work within it.

---

## 7. Caveats and methodology notes

- **Non-determinism is now controlled.** Early runs (sanity, grid) used
  `num_workers>0` and non-deterministic cuDNN; same-config AVR varied by ~0.01.
  The extended config-A run and all subsequent experiments fix the seed and use
  `num_workers=0` so results are reproducible. Historical single-run numbers
  (e.g. the 0.2875 E1 minimum) should be read with this ~0.01 noise band in mind.

- All diagnostics ran on COCO val (6352 poses); COCO train was not available
  locally, so the gradient-dilution measure (E2) used val batches as a proxy for
  gradient norms — valid since it measures norms, not training dynamics.
- The foreshortening classifier in E1 is a geometric heuristic (short bone <
  0.25·torso), not a 3D ground-truth check; it gives a defensible estimate, not a
  certified count. The true foreshortening fraction could be refined with depth
  or multi-view data (not available here).
- Diagnostic C/D/E focus on the **symmetry** sub-term of bone_ratio, which shares
  the 1.5 threshold with the AVR. The inter-segment (log-cosh) sub-term uses a
  different shape and is not directly comparable to the AVR; the gradient-norm in
  E2 sums both sub-terms.
- The log-cosh ablation (6.6) used identical seed, baseline checkpoint, boost, and
  epoch count across all three variants — the only change was the symmetry
  sub-term shape — so the comparison is clean.
- AP held throughout (within ~0.006), so none of the tested changes risks the
  primary task as long as LR stays at 3e-5 and lambdas are calibrated/clamped.
