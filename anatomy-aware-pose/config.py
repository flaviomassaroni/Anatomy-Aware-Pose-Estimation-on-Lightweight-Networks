"""Globals: percorsi dataset, iperparametri, seed, device.
Sezione 'Globals' della struttura richiesta dalla prof.

AUTO-DETECT LOCALE vs KAGGLE
----------------------------
Un solo file, nessun duplicato. I path si scelgono a runtime:
  - se accanto a questo config esiste una cartella ./datasets/  -> LOCALE
    (5070Ti): path ancorati a _HERE, portabili, niente /kaggle/... hardcodato.
  - altrimenti -> KAGGLE: path dei dataset montati in /kaggle/input.

Cosi lo stesso config.py committato su Git funziona in entrambi gli ambienti
e gli iperparametri (STL_BETA, LR, ...) NON possono piu' divergere tra due file.
Override manuale: esporta POSE_ENV=local oppure POSE_ENV=kaggle per forzare.
"""
import os
import random
import numpy as np
import torch

# --- Ancora: cartella che contiene questo config.py ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "datasets")

# --- Rilevamento ambiente ---
# Priorita': override esplicito via env var, poi presenza di ./datasets/,
# poi presenza di /kaggle/input. Default prudente: kaggle.
_forced = os.environ.get("POSE_ENV", "").strip().lower()
if _forced in ("local", "kaggle"):
    IS_LOCAL = (_forced == "local")
elif os.path.isdir(_DATA):
    IS_LOCAL = True
elif os.path.isdir("/kaggle/input"):
    IS_LOCAL = False
else:
    IS_LOCAL = False  # fallback: assume kaggle (path Kaggle, falliranno se assenti)

ENV_NAME = "LOCAL" if IS_LOCAL else "KAGGLE"

# --- Riproducibilita' (seed fissi + training deterministico) ---
SEED = 42

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Parametri pipeline (identici nei due ambienti) ---
NUM_KEYPOINTS = 17           # keypoint standard COCO
INPUT_SIZE    = (256, 192)   # (H, W) - standard pose estimation
HEATMAP_SIZE  = (64, 48)     # heatmap a 1/4 della risoluzione input
SIGMA         = 2            # ampiezza gaussiana nelle heatmap

# --- Path dataset: scelti in base all'ambiente ---
if IS_LOCAL:
    # LOCALE (5070Ti): tutto ancorato a ./datasets/ accanto ai moduli .py
    COCO_ROOT      = os.path.join(_DATA, "coco2017")
    COCO_TRAIN_IMG = os.path.join(COCO_ROOT, "train2017")
    COCO_VAL_IMG   = os.path.join(COCO_ROOT, "val2017")
    COCO_TRAIN_ANN = os.path.join(COCO_ROOT, "annotations", "person_keypoints_train2017.json")
    COCO_VAL_ANN   = os.path.join(COCO_ROOT, "annotations", "person_keypoints_val2017.json")

    OCHUMAN_IMG      = os.path.join(_DATA, "OCHuman", "images")
    OCHUMAN_VAL_ANN  = os.path.join(_DATA, "OCHuman", "ochuman_coco_format_val_range_0.00_1.00.json")
    OCHUMAN_TEST_ANN = os.path.join(_DATA, "OCHuman", "ochuman_coco_format_test_range_0.00_1.00.json")

    CHECKPOINT_DIR = os.path.join(_HERE, "checkpoints")
    BEST_PTH       = os.path.join(_HERE, "models", "best.pth")
else:
    # KAGGLE: dataset montati in /kaggle/input
    COCO_ROOT      = '/kaggle/input/datasets/asad11914/coco-2017-keypoints/coco2017'
    COCO_TRAIN_IMG = f'{COCO_ROOT}/train2017'
    COCO_VAL_IMG   = f'{COCO_ROOT}/val2017'
    COCO_TRAIN_ANN = f'{COCO_ROOT}/annotations/person_keypoints_train2017.json'
    COCO_VAL_ANN   = f'{COCO_ROOT}/annotations/person_keypoints_val2017.json'

    # ATTENZIONE: dopo aver montato OCHuman nel runner, verifica lo slug reale
    # con la cella os.walk. Se il path non corrisponde, correggi QUI e fai push.
    OCHUMAN_IMG      = '/kaggle/input/datasets/messinaalberto/ochuman/images/images'
    OCHUMAN_VAL_ANN  = '/kaggle/input/datasets/messinaalberto/ochuman/ochuman_coco_format_val_range_0.00_1.00.json'
    OCHUMAN_TEST_ANN = '/kaggle/input/datasets/messinaalberto/ochuman/ochuman_coco_format_test_range_0.00_1.00.json'

    CHECKPOINT_DIR = '/kaggle/working/checkpoints'
    # Su Kaggle il baseline checkpoint sta nel dataset condiviso; la cella del
    # notebook ha gia' il fallback a quel path se BEST_PTH non esiste.
    BEST_PTH       = '/kaggle/input/datasets/messinaalberto/pose-baseline-checkpoint/best.pth'

# --- STL fine-tuning (IDENTICI nei due ambienti: la sorgente di verita') ---
BONE_RATIO_THRESHOLD = 1.5    # soglia max/min per simmetria ossea (AVR + STL)
COLLAPSE_THRESHOLD   = 0.10   # segmento/torso minimo (AVR + STL)
MIN_CONF             = 0.3    # confidenza minima per l'AVR (score heatmap)
AVR_ANGLE_MIN_DEG    = 20.0   # floor angolare condiviso AVR e STL
STL_FINE_TUNE_LR     = 3e-5   # era 1e-4: ridotto per evitare catastrophic forgetting da E04
STL_TARGET_FRAC      = 0.1    # rho: frazione della spinta heatmap a cui portare ogni termine
STL_NUM_EPOCHS       = 10
STL_BETA             = 50     # beta=50: mediana gap 0.27 px, p95 3.2 (vs beta=30 p95=9.3); vedi analisi decoder gap