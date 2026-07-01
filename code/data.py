"""Data: parsing annotazioni COCO/OCHuman e Dataset PyTorch. Sezione 'Data'.
Resize aspect-ratio preserving (scala uniforme + padding centrato): la STL
ragiona su rapporti ossei, lo stretch li altererebbe.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import cv2

from utils import generate_heatmap

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


class COCOKeypointsDataset(Dataset):
    """Dataset di TRAINING: ritorna (immagine [3,256,192], heatmap [17,64,48], weight [17,1])."""

    def __init__(self, samples, img_dir, input_size, heatmap_size, sigma, num_kpts):
        self.samples = samples
        self.img_dir = img_dir
        self.input_h, self.input_w = input_size
        self.hm_h, self.hm_w = heatmap_size
        self.sigma = sigma
        self.num_kpts = num_kpts

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

        for i in range(self.num_kpts):
            kx, ky, v = kpts[i]
            if v == 0:                       # keypoint non annotato -> heatmap vuota
                target_weight[i] = 0.0
                continue
            cx = ((kx - x) * scale + pad_left) * hm_scale_x
            cy = ((ky - y) * scale + pad_top) * hm_scale_y
            if cx < 0 or cx >= self.hm_w or cy < 0 or cy >= self.hm_h:
                target_weight[i] = 0.0
                continue
            heatmaps[i] = generate_heatmap(cx, cy, self.hm_h, self.hm_w, self.sigma)

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
