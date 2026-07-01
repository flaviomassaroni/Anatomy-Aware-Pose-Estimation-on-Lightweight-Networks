"""Skeletal Topology Loss (STL): 4 termini differenziabili su coordinate 2D.

  1. bone_ratio         — rapporti inter-segmentali e simmetria sx/dx (Winter 2009)
  2. joint_angle        — angoli articolari fuori range fisiologico (AAOS 1965)
  3. geometric_ordering — giunti fuori ordine lungo le catene cinematiche
  4. collapse           — segmenti collassati rispetto alla scala del torso

Coordinate estratte dalle heatmap via soft-argmax (differenziabile).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import BONE_RATIO_THRESHOLD, COLLAPSE_THRESHOLD, SEVERITY_ALPHA, STL_BETA
from anthropometric_constraints import (
    BONE_RATIOS, SYMMETRY_PAIRS, JOINT_ANGLE_RANGES, KINEMATIC_CHAINS,
)


# Scala log-cosh per bone_ratio: 1.35 = log(1/cos(75°)), margine per foreshortening 2D.
BONE_SCALE = 1.35

# Severity weighting simmetria sx/dx: penalty *= (1 + SEVERITY_ALPHA * excess).
# Rinforza gradiente sui violatori correggibili (~17% batch), corregge diluizione.
# SEVERITY_ALPHA=0 → hinge pura (per ablation).

def _logcosh(x):
    """log(cosh(x)) stabile: L2 vicino a 0, L1 per |x| grande (robusto agli outlier)."""
    return x + F.softplus(-2.0 * x) - math.log(2.0)


# Estrae coordinate come valor medio pesato da softmax(beta * heatmap) (integral regression).
# beta = STL_BETA = 50: gap mediano 0.27 px con sigma=2. 
def soft_argmax(heatmaps, beta=STL_BETA):
    """Coordinate differenziabili [B,K,2] da heatmap [B,K,H,W]."""
    B, K, H, W = heatmaps.shape
    flat = heatmaps.reshape(B, K, -1)
    probs = F.softmax(beta * flat, dim=-1)
    probs = probs.reshape(B, K, H, W)
    device = heatmaps.device
    grid_x = torch.arange(W, dtype=torch.float32, device=device)
    grid_y = torch.arange(H, dtype=torch.float32, device=device)
    x = (probs.sum(dim=2) * grid_x).sum(dim=-1)
    y = (probs.sum(dim=3) * grid_y).sum(dim=-1)
    return torch.stack([x, y], dim=-1)


def _bone_length(coords, kp_a, kp_b):
    """Distanza euclidea [B] tra kp_a e kp_b. +1e-6 dentro sqrt: gradiente finito a dist=0."""
    diff = coords[:, kp_a, :] - coords[:, kp_b, :]
    return torch.sqrt((diff ** 2).sum(dim=-1) + 1e-6)


def _joint_angle(coords, kp_a, kp_joint, kp_b):
    """Angolo [B] in radianti al vertice kp_joint, range [0, pi].
    Usa atan2 invece di arccos: derivata stabile agli estremi 0 e pi.
    """
    v1 = coords[:, kp_a, :] - coords[:, kp_joint, :]
    v2 = coords[:, kp_b, :] - coords[:, kp_joint, :]
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    dot = (v1 * v2).sum(dim=-1)
    return torch.atan2(cross.abs(), dot)


def _kp_valid(valid_mask, *kp_indices):
    """[B] mask: 1.0 se tutti i keypoint indicati sono validi."""
    mask = torch.ones(valid_mask.shape[0], device=valid_mask.device)
    for kp in kp_indices:
        mask = mask * valid_mask[:, kp]
    return mask


def _masked_mean(per_sample_loss, mask):
    """Media pesata dalla mask. Ritorna 0 se nessun sample valido."""
    return (per_sample_loss * mask).sum() / (mask.sum() + 1e-6)


# (a) Rapporti inter-segmentali: log-cosh centrato sul nominale Winter. 3 regole × 2 lati.
# (b) Simmetria sx/dx: hinge + severity weighting. 4 coppie. 
def bone_ratio_loss(coords, valid_mask):
    """Penalizza rapporti ossei non plausibili: inter-segmentali (a) e simmetria sx/dx (b).
    coords: [B,K,2]; valid_mask: [B,K]. Ritorna: scalare.
    """
    losses = []

    # (a) Rapporti inter-segmentali: log-cosh (3 regole × lato sx + dx)
    for name, rule in BONE_RATIOS.items():
        nominal = rule['nominal']
        log_nom = math.log(nominal)
        num_a, num_b = rule['numerator']
        den_a, den_b = rule['denominator']
        for side_offset in [0, 1]:  # 0=sinistro, 1=destro (indici COCO: dx = sx+1)
            na, nb = num_a + side_offset, num_b + side_offset
            da, db = den_a + side_offset, den_b + side_offset
            mask = _kp_valid(valid_mask, na, nb, da, db)
            len_num = _bone_length(coords, na, nb)
            len_den = _bone_length(coords, da, db)
            ratio = len_num / (len_den + 1e-6)
            log_ratio = torch.log(ratio + 1e-6)
            z = (log_ratio - log_nom) / BONE_SCALE
            losses.append(_masked_mean(_logcosh(z), mask))

    # (b) Simmetria sx/dx: hinge su |log(ratio)|, soglia = log(BONE_RATIO_THRESHOLD)
    LOG_SYM_THRESHOLD = math.log(BONE_RATIO_THRESHOLD)
    SYMMETRY_CAP = 4.0
    for (left_a, left_b), (right_a, right_b), _ in SYMMETRY_PAIRS:
        mask = _kp_valid(valid_mask, left_a, left_b, right_a, right_b)
        len_left  = _bone_length(coords, left_a,  left_b)
        len_right = _bone_length(coords, right_a, right_b)
        ratio = len_left / (len_right + 1e-6)
        log_ratio = torch.log(ratio + 1e-6)
        excess = F.relu(log_ratio.abs() - LOG_SYM_THRESHOLD)
        base = torch.clamp(excess ** 2, max=SYMMETRY_CAP)
        penalty = base * (1.0 + SEVERITY_ALPHA * excess)
        losses.append(_masked_mean(penalty, mask))

    if not losses:
        return torch.tensor(0.0, device=coords.device)
    return torch.stack(losses).mean()


def joint_angle_loss(coords, valid_mask):
    """Penalizza angoli articolari fuori range fisiologico [deg_min, deg_max].
    coords: [B,K,2]; valid_mask: [B,K]. Ritorna: scalare.
    """
    losses = []
    for name, rule in JOINT_ANGLE_RANGES.items():
        kp_a, kp_joint, kp_b = rule['joints']
        deg_min, deg_max = rule['range_deg']
        mask = _kp_valid(valid_mask, kp_a, kp_joint, kp_b)
        rad_min = math.radians(deg_min)
        rad_max = math.radians(deg_max)
        angle = _joint_angle(coords, kp_a, kp_joint, kp_b)
        below = F.relu(rad_min - angle)
        above = F.relu(angle - rad_max)
        losses.append(_masked_mean(below ** 2 + above ** 2, mask))
    if not losses:
        return torch.tensor(0.0, device=coords.device)
    return torch.stack(losses).mean()


def collapse_loss(coords, valid_mask):
    """Penalizza segmenti collassati: dist_segmento/torso < COLLAPSE_THRESHOLD.
    Replica la categoria 'collapse' dell'AVR (stessi 4 giunti, stessa scala torso).
    """
    SHOULDER_L, SHOULDER_R, HIP_L, HIP_R = 5, 6, 11, 12
    COLLAPSE_JOINTS = [
        (5, 7, 9),    # spalla sx → gomito sx → polso sx
        (6, 8, 10),   # spalla dx → gomito dx → polso dx
        (11, 13, 15), # anca sx → ginocchio sx → caviglia sx
        (12, 14, 16), # anca dx → ginocchio dx → caviglia dx
    ]
    torso_kp_mask = _kp_valid(valid_mask, SHOULDER_L, SHOULDER_R, HIP_L, HIP_R)
    shoulder_mid = (coords[:, SHOULDER_L, :] + coords[:, SHOULDER_R, :]) / 2
    hip_mid      = (coords[:, HIP_L, :]      + coords[:, HIP_R, :])      / 2
    torso = torch.sqrt(((shoulder_mid - hip_mid) ** 2).sum(dim=-1) + 1e-6)
    losses = []
    for kp_a, kp_joint, kp_b in COLLAPSE_JOINTS:
        mask = torso_kp_mask * _kp_valid(valid_mask, kp_a, kp_joint, kp_b)
        d1 = _bone_length(coords, kp_a,    kp_joint) / (torso + 1e-6)
        d2 = _bone_length(coords, kp_joint, kp_b)    / (torso + 1e-6)
        penalty = (F.relu(COLLAPSE_THRESHOLD - d1) ** 2
                 + F.relu(COLLAPSE_THRESHOLD - d2) ** 2)
        losses.append(_masked_mean(penalty, mask))
    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=coords.device)


# Prior soft sul kinematic tree: giunto intermedio tra gli estremi. Non e' KPI dell'AVR.
def geometric_ordering_loss(coords, valid_mask):
    """Penalizza giunti fuori ordine lungo le catene cinematiche.
    t = dot(mid-a, b-a) / |b-a|^2; violazione se t < 0 o t > 1.
    """
    losses = []
    for kp_a, kp_mid, kp_b, _ in KINEMATIC_CHAINS:
        mask = _kp_valid(valid_mask, kp_a, kp_mid, kp_b)
        a   = coords[:, kp_a, :]
        mid = coords[:, kp_mid, :]
        b   = coords[:, kp_b, :]
        ab = b - a
        am = mid - a
        t = (am * ab).sum(dim=-1) / ((ab ** 2).sum(dim=-1) + 1e-6)
        below = F.relu(-t)
        above = F.relu(t - 1.0)
        losses.append(_masked_mean(below ** 2 + above ** 2, mask))
    if not losses:
        return torch.tensor(0.0, device=coords.device)
    return torch.stack(losses).mean()


class SkeletalTopologyLoss(nn.Module):
    """Loss combinata: L_hm + lambda_bone*L_bone + lambda_angle*L_angle
    + lambda_order*L_order + lambda_collapse*L_collapse.
    Lambda calibrati via calibrate_lambdas() prima del training.
    """

    def __init__(self, heatmap_criterion,
                 lambda_bone=0.5, lambda_angle=0.5, lambda_order=0.5,
                 lambda_collapse=0.5, beta=STL_BETA):
        super().__init__()
        self.heatmap_criterion = heatmap_criterion
        self.lambda_bone = lambda_bone
        self.lambda_angle = lambda_angle
        self.lambda_order = lambda_order
        self.lambda_collapse = lambda_collapse
        self.beta = beta

    def forward(self, pred_heatmaps, target_heatmaps, target_weight):
        """Ritorna (loss_totale, dict_termini) per logging.
        target_weight [B,K,1]: maschera keypoint non annotati (v=0 in COCO).
        """
        L_hm = self.heatmap_criterion(pred_heatmaps, target_heatmaps, target_weight)
        coords = soft_argmax(pred_heatmaps, beta=self.beta)
        # Masking via target_weight (GT), non score predetto: stabile in training.
        # Scelta deliberata rispetto all'AVR (MIN_CONF). 
        valid_mask = target_weight.squeeze(-1)
        L_bone     = bone_ratio_loss(coords, valid_mask)
        L_angle    = joint_angle_loss(coords, valid_mask)
        L_order    = geometric_ordering_loss(coords, valid_mask)
        L_collapse = collapse_loss(coords, valid_mask)
        L_total = (L_hm 
                   + self.lambda_bone    * L_bone
                   + self.lambda_angle   * L_angle
                   + self.lambda_order   * L_order
                   + self.lambda_collapse * L_collapse)
        terms = {
            'heatmap':  L_hm.item(),
            'bone':     L_bone.item(),
            'angle':    L_angle.item(),
            'order':    L_order.item(),
            'collapse': L_collapse.item(),
            'total':    L_total.item(),
        }
        return L_total, terms


# Calibra i lambda equalizzando norme di gradiente sul final_layer (GradNorm-style, statico).
# lambda_t = target_frac * ||grad L_hm|| / ||grad L_t||.
def _grad_norm_on(params):
    """Norma L2 del gradiente accumulato su una lista di parametri."""
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum().item())
    return total ** 0.5


@torch.no_grad()
def _zero_grads(model):
    for p in model.parameters():
        p.grad = None


def _clamp_lambda_spread(lambdas, max_spread):
    """Comprime lo spread dei lambda: max_ratio <= max_spread attorno alla media geometrica."""
    keys = list(lambdas.keys())
    log_vals = [math.log(lambdas[k] + 1e-12) for k in keys]
    center = sum(log_vals) / len(log_vals)
    max_dev = math.log(max_spread) / 2.0
    return {k: math.exp(center + max(-max_dev, min(max_dev, lv - center)))
            for k, lv in zip(keys, log_vals)}


def calibrate_lambdas(criterion, model, loader, device,
                      target_frac=0.1, n_batches=4, eps=1e-9,
                      max_spread=20.0, verbose=True):
    """Calibra i 4 lambda della STL su norma del gradiente (statico, 4 batch).
    Aggiorna criterion in-place. Ritorna dict {'bone','angle','order','collapse'}.
    """
    ref_params = list(model.head.final_layer.parameters())
    acc = {'heatmap': 0.0, 'bone': 0.0, 'angle': 0.0, 'order': 0.0, 'collapse': 0.0}
    seen = 0
    model.train()
    it = iter(loader)
    for _ in range(n_batches):
        try:
            imgs, hms, w = next(it)
        except StopIteration:
            break
        imgs, hms, w = imgs.to(device), hms.to(device), w.to(device)
        out = model(imgs)
        coords = soft_argmax(out, beta=criterion.beta)
        valid_mask = w.squeeze(-1)
        term_fns = {
            'heatmap':  lambda: criterion.heatmap_criterion(out, hms, w),
            'bone':     lambda: bone_ratio_loss(coords, valid_mask),
            'angle':    lambda: joint_angle_loss(coords, valid_mask),
            'order':    lambda: geometric_ordering_loss(coords, valid_mask),
            'collapse': lambda: collapse_loss(coords, valid_mask),
        }
        # Backward isolato per ogni termine: azzera grad, backward, leggi norma.
        # retain_graph perche' tutti i termini condividono lo stesso grafo (out, coords).
        keys = list(term_fns.keys())
        for i, k in enumerate(keys):
            _zero_grads(model)
            L = term_fns[k]()
            L.backward(retain_graph=(i < len(keys) - 1))
            acc[k] += _grad_norm_on(ref_params)
        seen += 1
    _zero_grads(model)
    for k in acc:
        acc[k] /= max(seen, 1)
    g_hm = acc['heatmap']
    lambdas = {}
    for k in ['bone', 'angle', 'order', 'collapse']:
        lambdas[k] = target_frac * g_hm / (acc[k] + eps)
    lambdas = _clamp_lambda_spread(lambdas, max_spread)
    criterion.lambda_bone     = lambdas['bone']
    criterion.lambda_angle    = lambdas['angle']
    criterion.lambda_order    = lambdas['order']
    criterion.lambda_collapse = lambdas['collapse']
    if verbose:
        print(f"Calibrazione lambda (gradient-norm, rho={target_frac}, "
              f"{seen} batch, ref=final_layer):")
        print(f"  norme grad grezze: " + "  ".join(
            f"{k}={acc[k]:.3e}" for k in ['heatmap', 'bone', 'angle', 'order', 'collapse']))
        print(f"  lambda calibrati : " + "  ".join(
            f"{k}={lambdas[k]:.5f}" for k in ['bone', 'angle', 'order', 'collapse']))
    return lambdas
