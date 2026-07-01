"""Data: parsing annotazioni COCO/OCHuman e Dataset PyTorch.
Sezione 'Data' della struttura richiesta dalla prof.

Nota: usa aspect-ratio preserving resize (scala uniforme + padding centrato),
NON stretch, perche' la STL ragiona su rapporti ossei e angoli.
"""
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO

from utils import generate_heatmap

import cv2
cv2.setNumThreads(0)  # evita thread contention con DataLoader num_workers > 0


def build_samples(ann_file, min_keypoints=1, min_bbox_area=32 * 32):
    """Una persona = un esempio. Filtra pose con troppo pochi keypoint o bbox degeneri.
    Funziona sia su COCO che su OCHuman (entrambi in formato COCO-style).
    """
    coco = COCO(ann_file)
    person_cat = coco.getCatIds(catNms=['person'])
    img_ids = coco.getImgIds(catIds=person_cat)

    samples = []
    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=person_cat)
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            if ann.get('num_keypoints', 0) < min_keypoints:
                continue
            x, y, w, h = ann['bbox']
            if w * h < min_bbox_area or w <= 0 or h <= 0:
                continue
            samples.append({
                'image_id':  img_id,
                'file_name': img_info['file_name'],
                'bbox':      ann['bbox'],       # [x, y, w, h]
                'keypoints': ann['keypoints'],  # 51 valori (17 * 3)
            })
    return samples


def _crop_and_pad(img, bbox, input_h, input_w):
    """Crop del bbox -> resize uniforme -> padding centrato su canvas (input_h, input_w).
    Ritorna il canvas e i parametri (scale, pad) per riusarli a valle se serve.
    """
    x, y, w, h = [int(v) for v in bbox]
    crop = img[y:y + h, x:x + w]
    scale = min(input_w / w, input_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    crop_resized = cv2.resize(crop, (new_w, new_h))
    pad_top = (input_h - new_h) // 2
    pad_left = (input_w - new_w) // 2
    canvas = np.zeros((input_h, input_w, 3), dtype=np.uint8)
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = crop_resized
    return canvas, (x, y, w, h), scale, pad_top, pad_left


# ===================================================================
# OCCLUSION AUGMENTATION (Pytel vs Han)
# ===================================================================
#
# Tre modalita', selezionabili a runtime, per studiare se l'occlusione
# sintetica in training migliora la robustezza su dati occlusi (OCHuman):
#
#   'random'       -> rettangolo casuale ovunque sul canvas (ipotesi Pytel:
#                     l'aug generica NON aiuta sotto occlusione).
#   'limb_segment' -> rettangolo orientato lungo un OSSO (es. avambraccio):
#                     occlusione strutturata "a segmento" (Han).
#   'limb_joint'   -> patch quadrata attorno a un GIUNTO (es. gomito):
#                     occlusione strutturata "a giunto" (Han).
#
# PRINCIPIO CHIAVE: si occlude SOLO l'immagine (canvas RGB). Le heatmap target
# e i target_weight restano INTATTI -> i keypoint coperti restano ground-truth
# SUPERVISIONATA. E' cosi' che il modello impara a INFERIRE il keypoint occluso
# dal contesto anatomico, invece di limitarsi a leggerne il pixel. Azzerare il
# target li renderebbe inutili: l'aug non insegnerebbe nulla.
#
# Le coordinate keypoint usate qui sono gia' in spazio CANVAS (post crop+pad).

# bone segments (COCO): coppie (prossimale, distale) per limb_segment
_AUG_BONES = [(5, 7), (7, 9), (6, 8), (8, 10),      # braccia: spalla-gomito, gomito-polso
              (11, 13), (13, 15), (12, 14), (14, 16)]  # gambe: anca-ginocchio, ginocchio-caviglia
# giunti occludibili (COCO) per limb_joint
_AUG_JOINTS = [7, 9, 8, 10, 13, 15, 14, 16, 5, 6, 11, 12]


def _occlude_random(canvas, rng):
    """Rettangolo nero casuale (Pytel). Lato 20-45% della dimensione del canvas."""
    H, W = canvas.shape[:2]
    rw = int(W * rng.uniform(0.20, 0.45))
    rh = int(H * rng.uniform(0.20, 0.45))
    x0 = rng.integers(0, max(1, W - rw))
    y0 = rng.integers(0, max(1, H - rh))
    canvas[y0:y0 + rh, x0:x0 + rw] = 0
    return canvas


def _occlude_limb_segment(canvas, kps_canvas, vis, rng):
    """Occlude un osso a caso (tra quelli con ENTRAMBI gli estremi visibili)
    con un rettangolo orientato lungo il segmento (spessore ~ lunghezza osso)."""
    H, W = canvas.shape[:2]
    avail = [(a, b) for (a, b) in _AUG_BONES if vis[a] and vis[b]]
    if not avail:
        return canvas
    a, b = avail[rng.integers(len(avail))]
    pa, pb = kps_canvas[a], kps_canvas[b]
    length = float(np.hypot(*(pb - pa)) + 1e-6)
    thick = max(6, int(0.45 * length))      # spessore proporzionale all'osso
    # poligono orientato: rettangolo lungo il segmento a-b
    d = (pb - pa) / length
    n = np.array([-d[1], d[0]])             # normale
    half = thick / 2.0
    pts = np.array([pa + n*half, pb + n*half, pb - n*half, pa - n*half], dtype=np.int32)
    cv2.fillConvexPoly(canvas, pts, 0)
    return canvas


def _occlude_limb_joint(canvas, kps_canvas, vis, rng):
    """Occlude un giunto a caso (tra quelli visibili) con una patch quadrata."""
    H, W = canvas.shape[:2]
    avail = [j for j in _AUG_JOINTS if vis[j]]
    if not avail:
        return canvas
    j = avail[rng.integers(len(avail))]
    cx, cy = kps_canvas[j]
    side = int(min(H, W) * rng.uniform(0.18, 0.30))
    x0, y0 = int(cx - side/2), int(cy - side/2)
    x1, y1 = x0 + side, y0 + side
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    canvas[y0:y1, x0:x1] = 0
    return canvas


def apply_occlusion(canvas, kps_canvas, vis, aug_mode, aug_prob, aug_num, rng):
    """Applica l'occlusione al solo canvas. Ritorna il canvas modificato.

    canvas:     [H,W,3] uint8 (post crop+pad)
    kps_canvas: [K,2] coordinate keypoint in spazio canvas
    vis:        [K] bool, True se il keypoint e' annotato/valido
    aug_mode:   'none' | 'random' | 'limb_segment' | 'limb_joint'
    aug_prob:   probabilita' di applicare l'occlusione a questa immagine
    aug_num:    quante occlusioni applicare (se attiva)
    rng:        np.random.Generator (per riproducibilita')
    """
    if aug_mode == 'none' or rng.random() > aug_prob:
        return canvas
    for _ in range(max(1, aug_num)):
        if aug_mode == 'random':
            canvas = _occlude_random(canvas, rng)
        elif aug_mode == 'limb_segment':
            canvas = _occlude_limb_segment(canvas, kps_canvas, vis, rng)
        elif aug_mode == 'limb_joint':
            canvas = _occlude_limb_joint(canvas, kps_canvas, vis, rng)
    return canvas


class COCOKeypointsDataset(Dataset):
    """Dataset di TRAINING: ritorna (immagine [3,256,192], heatmap [17,64,48], weight [17,1]).

    Occlusion augmentation (solo training): passa aug_mode != 'none' per attivarla.
    La GT (heatmap + target_weight) resta SEMPRE intatta: si occlude solo l'immagine.
    """

    def __init__(self, samples, img_dir, input_size, heatmap_size, sigma, num_kpts,
                 aug_mode='none', aug_prob=0.5, aug_num=1, seed=42):
        self.samples = samples
        self.img_dir = img_dir
        self.input_h, self.input_w = input_size
        self.hm_h, self.hm_w = heatmap_size
        self.sigma = sigma
        self.num_kpts = num_kpts
        # --- occlusion augmentation (solo training) ---
        self.aug_mode = aug_mode      # 'none'|'random'|'limb_segment'|'limb_joint'
        self.aug_prob = aug_prob      # prob. di occludere un'immagine
        self.aug_num  = aug_num       # quante occlusioni per immagine
        self._aug_seed = seed         # base per RNG per-campione (riproducibile)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = cv2.imread(os.path.join(self.img_dir, s['file_name']))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        canvas, (x, y, w, h), scale, pad_top, pad_left = _crop_and_pad(
            img, s['bbox'], self.input_h, self.input_w
        )

        kpts = np.array(s['keypoints']).reshape(self.num_kpts, 3)
        heatmaps = np.zeros((self.num_kpts, self.hm_h, self.hm_w), dtype=np.float32)
        target_weight = np.ones((self.num_kpts, 1), dtype=np.float32)

        hm_scale_x = self.hm_w / self.input_w
        hm_scale_y = self.hm_h / self.input_h

        # coordinate keypoint in spazio CANVAS (per l'occlusione strutturata)
        kps_canvas = np.zeros((self.num_kpts, 2), dtype=np.float32)
        vis = np.zeros(self.num_kpts, dtype=bool)

        for i in range(self.num_kpts):
            kx, ky, v = kpts[i]
            if v == 0:                       # keypoint non annotato -> heatmap vuota
                target_weight[i] = 0.0
                continue
            # spazio canvas (per occlusione) e spazio heatmap (per GT)
            canvas_x = (kx - x) * scale + pad_left
            canvas_y = (ky - y) * scale + pad_top
            kps_canvas[i] = (canvas_x, canvas_y)
            vis[i] = True
            cx = canvas_x * hm_scale_x
            cy = canvas_y * hm_scale_y
            if cx < 0 or cx >= self.hm_w or cy < 0 or cy >= self.hm_h:
                target_weight[i] = 0.0
                vis[i] = False
                continue
            heatmaps[i] = generate_heatmap(cx, cy, self.hm_h, self.hm_w, self.sigma)

        # --- OCCLUSION AUGMENTATION: solo immagine, GT (heatmap+weight) intatta ---
        # Le heatmap sono gia' calcolate sopra dai keypoint ORIGINALI; qui tocchiamo
        # solo il canvas RGB. I keypoint occlusi restano supervisionati (target_weight
        # invariato) -> il modello deve inferirli dal contesto. RNG seedato per
        # riproducibilita' deterministica (seed base + idx).
        if self.aug_mode != 'none':
            rng = np.random.default_rng(self._aug_seed + idx)
            canvas = apply_occlusion(canvas, kps_canvas, vis,
                                     self.aug_mode, self.aug_prob, self.aug_num, rng)

        img_tensor = torch.from_numpy(canvas).permute(2, 0, 1).float() / 255.0
        hm_tensor = torch.from_numpy(heatmaps).float()
        weight_tensor = torch.from_numpy(target_weight).float()
        return img_tensor, hm_tensor, weight_tensor


class COCOEvalDataset(Dataset):
    """Dataset di INFERENZA: ritorna (immagine [3,256,192], image_id, bbox [4]).
    Niente heatmap GT: la ground truth per AP/AR entra a parte via il file di annotazione.
    """

    def __init__(self, samples, img_dir, input_size):
        self.samples = samples
        self.img_dir = img_dir
        self.input_h, self.input_w = input_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = cv2.imread(os.path.join(self.img_dir, s['file_name']))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        canvas, (x, y, w, h), _, _, _ = _crop_and_pad(img, s['bbox'], self.input_h, self.input_w)
        img_tensor = torch.from_numpy(canvas).permute(2, 0, 1).float() / 255.0
        bbox_arr = np.array([x, y, w, h], dtype=np.float32)
        return img_tensor, s['image_id'], bbox_arr