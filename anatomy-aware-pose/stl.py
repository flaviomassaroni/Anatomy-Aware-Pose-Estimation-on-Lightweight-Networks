"""Skeletal Topology Loss (STL) — loss differenziabile con prior anatomici.

Tre termini:
  1. Bone ratio   — rapporti di lunghezza ossea fuori range antropometrico
  2. Joint angle  — angoli articolari fuori range fisiologico
  3. Geometric ordering — giunti fuori ordine lungo le catene cinematiche

Tutti i termini operano su COORDINATE, non su heatmap. Il ponte e' la
soft-argmax, che estrae coordinate differenziabili dalle heatmap.

Fonti: Winter 2009, Drillis & Contini 1966, AAOS 1965.
Vedi anthropometric_constraints.py per i range e le citazioni.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from anthropometric_constraints import (
    BONE_RATIOS, SYMMETRY_PAIRS, SYMMETRY_RANGE,
    JOINT_ANGLE_RANGES, KINEMATIC_CHAINS,
)


# ===================================================================
# SCALA DELLA PENALITA' BONE-RATIO (log-cosh)
# ===================================================================
#
# I rapporti inter-segmentali (avambraccio/braccio, gamba/coscia, ...)
# confrontano segmenti DIVERSI, che in proiezione 2D monoculare possono
# subire foreshortening INDIPENDENTE. Un rapporto 3D nominale (Winter)
# di 0.785 puo' quindi apparire, in 2D, ovunque tra ~0.20 e ~3.0 in modo
# perfettamente legittimo (un segmento puntato verso la camera a 75 gradi
# si proietta a cos(75 gradi) ~ 0.26 della lunghezza reale).
#
# Una hinge a confini netti penalizza tutto questo foreshortening come se
# fosse un errore -> esplode (L_bone ~ 37 nei nostri test) e domina il
# gradiente con pochi outlier. Usiamo invece una penalita' log-cosh
# centrata sul LOG del rapporto nominale di Winter:
#
#   pen(ratio) = logcosh( ( log(ratio) - log(nominale) ) / BONE_SCALE )
#
# Perche' lo spazio log: rende simmetrica la penalita' tra un rapporto e
# il suo reciproco (2x e 0.5x "sbagliano uguale"), coerente col fatto che
# il foreshortening accorcia e allunga con la stessa probabilita'.
#
# Perche' BONE_SCALE = 1.35: e' la distanza in spazio log dal nominale al
# confine di foreshortening geometricamente atteso. Per qualunque coppia,
# log(nom / (nom*cos75)) = log(1/cos75) = log(1/0.259) ~ 1.35. E' UN solo
# parametro geometrico (dipende solo da cos(75 gradi)), non tarato sul
# dataset: dentro l'intervallo di foreshortening atteso la penalita' resta
# trascurabile, oltre cresce. Vedi anthropometric_constraints.py 'range'
# (ora documentazione del foreshortening atteso, non piu' confini hinge).
BONE_SCALE = 1.35


def _logcosh(x):
    """log(cosh(x)) numericamente stabile (no overflow per |x| grande).

    Per |x| grande, log(cosh(x)) ~ |x| - log(2): si comporta come L1
    (robusto agli outlier). Per |x| piccolo ~ x^2/2: si comporta come L2
    (liscio, gradiente ben definito al centro).
    """
    return x + F.softplus(-2.0 * x) - math.log(2.0)


# ===================================================================
# SOFT-ARGMAX (differenziabile, sostituisce argmax in training)
# ===================================================================
#
# Riferimento: Sun et al., "Integral Human Pose Regression", ECCV 2018.
#
# Idea: data una heatmap h[H,W], la trattiamo come distribuzione di
# probabilita' (via softmax) e calcoliamo il valore atteso di (x, y).
#
# Formula:
#   p(i,j) = softmax(beta * h)        -- normalizza in [0,1], somma=1
#   x_hat  = sum_j  j * sum_i p(i,j)  -- media pesata delle colonne
#   y_hat  = sum_i  i * sum_j p(i,j)  -- media pesata delle righe
#
# beta (temperatura) controlla la nitidezza:
#   beta basso (~1)  -> distribuzione piatta, coordinate imprecise
#   beta alto (~100) -> quasi-argmax, gradienti che svaniscono
#   beta ~10         -> buon compromesso per heatmap con sigma=2

def soft_argmax(heatmaps, beta=10.0):
    """Estrae coordinate differenziabili da heatmap.

    Input:  heatmaps [B, K, H, W]  (output grezzo del modello)
    Output: coords   [B, K, 2]     (x, y in spazio heatmap)

    Perche' beta=10? Le nostre heatmap sono gaussiane con picco ~1.0 e
    sigma=2. Con beta=10, softmax(10 * 1.0) >> softmax(10 * 0.1), quindi
    il picco domina. Con beta=1, la differenza e' troppo piccola e le
    coordinate collassano verso il centro. Con beta=100, la softmax e'
    quasi un delta e i gradienti sono troppo piccoli. beta=10 e' il
    punto in cui le coordinate sono precise E i gradienti fluiscono.
    """
    B, K, H, W = heatmaps.shape

    # Softmax spaziale: [B, K, H*W] -> distribuzione di probabilita'
    flat = heatmaps.reshape(B, K, -1)            # [B, K, H*W]
    probs = F.softmax(beta * flat, dim=-1)        # [B, K, H*W]
    probs = probs.reshape(B, K, H, W)             # [B, K, H, W]

    # Griglie di coordinate (create una volta, riusate per tutto il batch)
    # grid_x = [0, 1, 2, ..., W-1],  grid_y = [0, 1, 2, ..., H-1]
    device = heatmaps.device
    grid_x = torch.arange(W, dtype=torch.float32, device=device)  # [W]
    grid_y = torch.arange(H, dtype=torch.float32, device=device)  # [H]

    # Valore atteso di x: somma su righe -> [B,K,W], poi prodotto con grid_x
    # Valore atteso di y: somma su colonne -> [B,K,H], poi prodotto con grid_y
    x = (probs.sum(dim=2) * grid_x).sum(dim=-1)   # [B, K]
    y = (probs.sum(dim=3) * grid_y).sum(dim=-1)   # [B, K]

    return torch.stack([x, y], dim=-1)             # [B, K, 2]


# ===================================================================
# Utility geometriche differenziabili
# ===================================================================

def _bone_length(coords, kp_a, kp_b):
    """Distanza euclidea tra due keypoint. [B] scalare per persona.

    coords: [B, K, 2]
    Ritorna: [B] lunghezze (sempre >= 0, differenziabile via sqrt+eps).
    """
    diff = coords[:, kp_a, :] - coords[:, kp_b, :]   # [B, 2]
    return torch.sqrt((diff ** 2).sum(dim=-1) + 1e-6)  # [B]
    # Il +1e-6 dentro la sqrt evita gradiente infinito quando dist->0.


def _joint_angle(coords, kp_a, kp_joint, kp_b):
    """Angolo al vertice kp_joint, in RADIANTI, range [0, pi].

    Usa atan2 invece di arccos per stabilita' numerica:
    - arccos ha derivata infinita a 0 e pi (i nostri estremi!)
    - atan2 e' stabile ovunque

    La formula: angle = atan2(|cross|, dot) dove
      cross = v1.x * v2.y - v1.y * v2.x   (prodotto vettore, scalare in 2D)
      dot   = v1 . v2                       (prodotto scalare)
    """
    v1 = coords[:, kp_a, :] - coords[:, kp_joint, :]      # [B, 2]
    v2 = coords[:, kp_b, :] - coords[:, kp_joint, :]      # [B, 2]

    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]     # [B]
    dot = (v1 * v2).sum(dim=-1)                             # [B]

    angle = torch.atan2(cross.abs(), dot)                   # [B], in [0, pi]
    # .abs() sul cross perche' ci interessa l'angolo non-orientato
    return angle


# ===================================================================
# TERMINE 1: Bone Ratio Loss
# ===================================================================
#
# Due sotto-termini:
#   a) Rapporti inter-segmentali (avambraccio/braccio, gamba/coscia, ...)
#      -> penalita' se il rapporto esce dal range [r_min, r_max]
#   b) Simmetria sx/dx (braccio sx vs dx, coscia sx vs dx, ...)
#      -> penalita' se il rapporto sx/dx esce dal range [0.65, 1.55]
#
# Penalita': hinge loss al quadrato (smooth, zero dentro il range):
#   penalty = max(0, r_min - ratio)^2 + max(0, ratio - r_max)^2

def _kp_valid(valid_mask, *kp_indices):
    """Ritorna [B] mask: 1.0 se TUTTI i keypoint indicati sono validi, 0.0 altrimenti."""
    mask = torch.ones(valid_mask.shape[0], device=valid_mask.device)
    for kp in kp_indices:
        mask = mask * valid_mask[:, kp]
    return mask


def _masked_mean(per_sample_loss, mask):
    """Media pesata dalla mask. Se nessun sample e' valido, ritorna 0."""
    return (per_sample_loss * mask).sum() / (mask.sum() + 1e-6)


def bone_ratio_loss(coords, valid_mask):
    """Penalizza rapporti ossei non plausibili.

    coords: [B, K, 2] coordinate da soft-argmax

    Due sotto-termini con forme DIVERSE, ciascuna giustificata dalla fisica:
      a) Rapporti inter-segmentali -> log-cosh sul log-rapporto centrato sul
         nominale di Winter. Confrontano segmenti diversi, soggetti a
         foreshortening indipendente -> serve una penalita' robusta che
         tolleri la varianza di proiezione (vedi BONE_SCALE sopra).
      b) Simmetria sx/dx -> hinge quadratica. Confronta lo STESSO segmento
         sui due lati: il foreshortening e' in gran parte condiviso (assunta
         co-planarita' approssimata dei due lati), quindi devia da 1 solo
         per errori di stima sotto occlusione -> la hinge ha senso fisico.
         (Limite noto: pose di profilo estremo rompono la co-planarita'; il
         margine largo del range [0.65, 1.55] assorbe la maggior parte dei
         casi. Da rivedere solo se la per_category AVR segnala un problema.)

    Ritorna: scalare (media su batch e su tutte le regole)
    """
    losses = []

    # --- a) Rapporti inter-segmentali: log-cosh (3 regole, sx E dx) ---
    # In anthropometric_constraints.py i keypoint sono definiti per il lato
    # sinistro. Per il destro, gli indici COCO sono sempre +1
    # (es. left_shoulder=5, right_shoulder=6).
    for name, rule in BONE_RATIOS.items():
        nominal = rule['nominal']
        log_nom = math.log(nominal)
        num_a, num_b = rule['numerator']
        den_a, den_b = rule['denominator']

        for side_offset in [0, 1]:  # 0=sinistro, 1=destro
            na, nb = num_a + side_offset, num_b + side_offset
            da, db = den_a + side_offset, den_b + side_offset
            mask = _kp_valid(valid_mask, na, nb, da, db)

            len_num = _bone_length(coords, na, nb)
            len_den = _bone_length(coords, da, db)
            ratio = len_num / (len_den + 1e-6)  # [B]

            # log-cosh centrato sul nominale di Winter, scala geometrica.
            # +1e-6 dentro il log evita log(0) se un segmento collassa.
            log_ratio = torch.log(ratio + 1e-6)
            z = (log_ratio - log_nom) / BONE_SCALE
            losses.append(_masked_mean(_logcosh(z), mask))

    # --- b) Simmetria sx/dx: hinge quadratica con clamp (4 regole) ---
    SYMMETRY_CAP = 4.0  # oltre il cap la posa e' irrecuperabile, satura
    s_min, s_max = SYMMETRY_RANGE
    for (left_a, left_b), (right_a, right_b), _ in SYMMETRY_PAIRS:
        mask = _kp_valid(valid_mask, left_a, left_b, right_a, right_b)

        len_left = _bone_length(coords, left_a, left_b)
        len_right = _bone_length(coords, right_a, right_b)
        ratio = len_left / (len_right + 1e-6)  # [B]

        below = F.relu(s_min - ratio)
        above = F.relu(ratio - s_max)
        penalty = torch.clamp(below ** 2 + above ** 2, max=SYMMETRY_CAP)
        losses.append(_masked_mean(penalty, mask))

    if not losses:
        return torch.tensor(0.0, device=coords.device)
    return torch.stack(losses).mean()


# ===================================================================
# TERMINE 2: Joint Angle Loss
# ===================================================================
#
# Per ogni giunto, calcola l'angolo 2D e penalizza se esce dal range
# fisiologico [theta_min, theta_max].
#
# Perche' atan2 e non arccos?
#   arccos(cos_theta) ha derivata -1/sqrt(1 - cos^2), che esplode
#   quando theta -> 0 o theta -> pi (esattamente i confini del range!).
#   atan2(|sin_theta|, cos_theta) ha derivata stabile ovunque.

def joint_angle_loss(coords, valid_mask):
    """Penalizza angoli articolari fuori range fisiologico.

    coords:     [B, K, 2] coordinate da soft-argmax
    valid_mask: [B, K]     1.0 per keypoint validi

    Ritorna: scalare (media su batch e su tutte le regole valide)
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


# ===================================================================
# TERMINE 3: Geometric Ordering Loss
# ===================================================================
#
# Per ogni catena (a -> mid -> b), il giunto intermedio deve stare
# "tra" a e b proiettato sulla direzione a->b.
#
# Calcolo: t = dot(mid - a, b - a) / dot(b - a, b - a)
#   t in [0, 1] -> mid sta tra a e b (OK)
#   t < 0       -> mid e' "prima" di a (violazione)
#   t > 1       -> mid e' "dopo" b (violazione)
#
# Penalita': max(0, -t)^2 + max(0, t - 1)^2

def geometric_ordering_loss(coords, valid_mask):
    """Penalizza giunti fuori ordine lungo le catene cinematiche.

    coords:     [B, K, 2] coordinate da soft-argmax
    valid_mask: [B, K]     1.0 per keypoint validi

    Ritorna: scalare (media su batch e catene valide)
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


# ===================================================================
# LOSS COMBINATA
# ===================================================================

class SkeletalTopologyLoss(nn.Module):
    """Loss combinata: L_heatmap + lambda_bone * L_bone
                                 + lambda_angle * L_angle
                                 + lambda_order * L_order

    I lambda controllano il peso di ogni termine. Valori di partenza
    suggeriti per il grid search: lambda ~ 0.1 - 1.0 ciascuno.
    Con lambda troppo alti la STL domina e l'AP crolla; con lambda
    troppo bassi la STL non ha effetto.
    """

    def __init__(self, heatmap_criterion,
                 lambda_bone=0.5, lambda_angle=0.5, lambda_order=0.5,
                 beta=10.0):
        super().__init__()
        self.heatmap_criterion = heatmap_criterion
        self.lambda_bone = lambda_bone
        self.lambda_angle = lambda_angle
        self.lambda_order = lambda_order
        self.beta = beta

    def forward(self, pred_heatmaps, target_heatmaps, target_weight):
        """
        pred_heatmaps:   [B, K, H, W]  output del modello
        target_heatmaps: [B, K, H, W]  ground truth gaussiane
        target_weight:   [B, K, 1]     1.0 per keypoint validi, 0.0 per mancanti

        Ritorna: loss_totale, dict con i singoli termini (per logging)
        """
        # 1. Loss sulle heatmap (identica alla baseline)
        L_hm = self.heatmap_criterion(pred_heatmaps, target_heatmaps, target_weight)

        # 2. Estrai coordinate differenziabili
        coords = soft_argmax(pred_heatmaps, beta=self.beta)  # [B, K, 2]

        # 3. Maschera di validita': [B, K]
        #    La STL opera SOLO sui keypoint annotati. Senza maschera,
        #    i keypoint con target_weight=0 hanno heatmap rumore ->
        #    coordinate casuali -> rapporti/angoli assurdi -> loss esplode.
        valid_mask = target_weight.squeeze(-1)  # [B, K]

        # 4. Termini STL (mascherati)
        L_bone  = bone_ratio_loss(coords, valid_mask)
        L_angle = joint_angle_loss(coords, valid_mask)
        L_order = geometric_ordering_loss(coords, valid_mask)

        # 4. Combinazione pesata
        L_total = (L_hm
                   + self.lambda_bone * L_bone
                   + self.lambda_angle * L_angle
                   + self.lambda_order * L_order)

        # Dict per logging (utile per monitorare quale termine domina)
        terms = {
            'heatmap': L_hm.item(),
            'bone':    L_bone.item(),
            'angle':   L_angle.item(),
            'order':   L_order.item(),
            'total':   L_total.item(),
        }
        return L_total, terms