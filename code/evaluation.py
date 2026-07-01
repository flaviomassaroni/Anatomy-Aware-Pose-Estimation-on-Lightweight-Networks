"""Evaluation: inferenza, AP/AR (pycocotools), AVR (custom), wrapper COCO/OCHuman.
"""
import json
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from config import (NUM_KEYPOINTS, INPUT_SIZE, HEATMAP_SIZE, COCO_VAL_IMG, COCO_VAL_ANN, OCHUMAN_IMG, OCHUMAN_VAL_ANN, BONE_RATIO_THRESHOLD, COLLAPSE_THRESHOLD, MIN_CONF, AVR_ANGLE_MIN_DEG, RESULTS_DIR)
from utils import decode_heatmaps, heatmap_to_original
from data import build_samples, COCOEvalDataset


def run_inference(model, samples, img_dir, device, batch_size=32, num_workers=0):
    """Gira il modello su tutti i sample. Ritorna:
    - coco_results: lista di dict in formato COCO results (per AP/AR)
    - coords_all, scores_all: per persona, coordinate in spazio immagine e confidenze (per AVR)
    """
    eval_dataset = COCOEvalDataset(samples, img_dir, INPUT_SIZE)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    model.eval()
    coco_results, coords_all, scores_all = [], [], []
    with torch.no_grad():
        for imgs, image_ids, bboxes in tqdm(eval_loader, desc="inferenza", leave=False):
            imgs = imgs.to(device)
            heatmaps = model(imgs)
            coords_hm, scores = decode_heatmaps(heatmaps)
            bboxes_np = bboxes.numpy()
            image_ids_np = image_ids.numpy()
            for i in range(imgs.shape[0]):
                coords_img = heatmap_to_original(coords_hm[i], bboxes_np[i], INPUT_SIZE, HEATMAP_SIZE)
                kp_scores = scores[i]
                keypoints_flat = []
                for k in range(NUM_KEYPOINTS):
                    keypoints_flat += [float(coords_img[k, 0]), float(coords_img[k, 1]), float(kp_scores[k])]
                coco_results.append({
                    'image_id': int(image_ids_np[i]),
                    'category_id': 1,
                    'keypoints': keypoints_flat,
                    'score': float(kp_scores.mean()),
                })
                coords_all.append(coords_img)
                scores_all.append(kp_scores)
    return coco_results, coords_all, scores_all


def run_coco_eval(coco_results, ann_file, results_path):
    with open(results_path, 'w') as f:
        json.dump(coco_results, f)
    coco_gt = COCO(ann_file)
    coco_dt = coco_gt.loadRes(results_path)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='keypoints')
    coco_eval.params.imgIds = sorted(list(set(r['image_id'] for r in coco_results)))
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval


KP_NAMES = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
KP_IDX = {name: i for i, name in enumerate(KP_NAMES)}


SYMMETRIC_BONE_PAIRS = [
    (('left_shoulder', 'left_elbow'), ('right_shoulder', 'right_elbow')),
    (('left_elbow', 'left_wrist'),    ('right_elbow', 'right_wrist')),
    (('left_hip', 'left_knee'),       ('right_hip', 'right_knee')),
    (('left_knee', 'left_ankle'),     ('right_knee', 'right_ankle')),
]
# (estremo1, giunto, estremo2, angolo minimo plausibile in gradi)
ANGLE_JOINTS = [
    ('left_hip', 'left_knee', 'left_ankle', AVR_ANGLE_MIN_DEG),
    ('right_hip', 'right_knee', 'right_ankle', AVR_ANGLE_MIN_DEG),
    ('left_shoulder', 'left_elbow', 'left_wrist', AVR_ANGLE_MIN_DEG),
    ('right_shoulder', 'right_elbow', 'right_wrist', AVR_ANGLE_MIN_DEG),
]



def _dist(p1, p2):
    return np.linalg.norm(p1 - p2)


def _angle(p_a, p_joint, p_b):
    """Angolo in [0, 180] gradi al vertice p_joint."""
    v1, v2 = p_a - p_joint, p_b - p_joint
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))


def compute_avr_violations(coords, scores, min_conf=MIN_CONF):
    """Quante regole anatomiche viola UNA posa."""
    def valid(name): return scores[KP_IDX[name]] >= min_conf
    def pt(name):    return coords[KP_IDX[name]]

    violations = {'bone_ratio': 0, 'joint_angle': 0, 'collapse': 0}

    # scala di riferimento per-persona (lunghezza del torso), per rendere i test scale-invariant
    torso_scale = None
    if all(valid(k) for k in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']):
        shoulder_mid = (pt('left_shoulder') + pt('right_shoulder')) / 2
        hip_mid = (pt('left_hip') + pt('right_hip')) / 2
        torso_scale = _dist(shoulder_mid, hip_mid)

    # simmetria sx/dx delle ossa
    for (a1, b1), (a2, b2) in SYMMETRIC_BONE_PAIRS:
        if valid(a1) and valid(b1) and valid(a2) and valid(b2):
            len1, len2 = _dist(pt(a1), pt(b1)), _dist(pt(a2), pt(b2))
            if min(len1, len2) > 1e-3 and max(len1, len2) / min(len1, len2) > BONE_RATIO_THRESHOLD:
                violations['bone_ratio'] += 1

    # angoli articolari sotto il minimo plausibile
    for a, joint, b, min_angle in ANGLE_JOINTS:
        if valid(a) and valid(joint) and valid(b):
            if _angle(pt(a), pt(joint), pt(b)) < min_angle:
                violations['joint_angle'] += 1

    # collasso di un segmento (giunto sovrapposto a un estremo)
    if torso_scale and torso_scale > 1e-3:
        for a, joint, b, _ in ANGLE_JOINTS:
            if valid(a) and valid(joint) and valid(b):
                d1 = _dist(pt(a), pt(joint)) / torso_scale
                d2 = _dist(pt(joint), pt(b)) / torso_scale
                if d1 < COLLAPSE_THRESHOLD or d2 < COLLAPSE_THRESHOLD:
                    violations['collapse'] += 1

    violations['total'] = sum(violations.values())
    violations['any'] = int(violations['total'] > 0)
    return violations


def evaluate_avr(coords_list, scores_list):
    all_v = [compute_avr_violations(c, s) for c, s in zip(coords_list, scores_list)]
    n = max(len(all_v), 1)
    return {
        'AVR_pose_rate': sum(v['any'] for v in all_v) / n,
        'AVR_mean_violations': sum(v['total'] for v in all_v) / n,
        'per_category': {cat: sum(v[cat] for v in all_v) / n
                         for cat in ['bone_ratio', 'joint_angle', 'collapse']},
        'n_poses': len(all_v),
    }


def evaluate_on_coco_val(model, val_samples, device,
                         results_path=os.path.join(RESULTS_DIR, 'coco_val_pred.json')):
    """Inferenza + AP/AR + AVR su COCO val. Ritorna (coco_eval, avr_dict)."""
    coco_results, coords_all, scores_all = run_inference(model, val_samples, COCO_VAL_IMG, device)
    coco_eval = run_coco_eval(coco_results, COCO_VAL_ANN, results_path)
    avr = evaluate_avr(coords_all, scores_all)
    return coco_eval, avr


def evaluate_on_ochuman(model, device, ann_file=OCHUMAN_VAL_ANN, img_dir=OCHUMAN_IMG,
                        results_path=os.path.join(RESULTS_DIR, 'ochuman_pred.json')):
    """Valutazione ZERO-SHOT su OCHuman (nessun fine-tuning), stesso protocollo GT-bbox.
    OCHuman e' in formato COCO -> riuso build_samples + run_inference + run_coco_eval.
    AP/AR usano le GT keypoints di OCHuman; l'AVR no (misura la coerenza interna).
    """
    oc_samples = build_samples(ann_file)
    coco_results, coords_all, scores_all = run_inference(model, oc_samples, img_dir, device)
    coco_eval = run_coco_eval(coco_results, ann_file, results_path)
    avr = evaluate_avr(coords_all, scores_all)
    return coco_eval, avr
