"""Globals: percorsi dataset, iperparametri, seed, device.
Sezione 'Globals' della struttura richiesta dalla prof.

VERSIONE LOCALE (5070Ti). I path sono ancorati alla posizione di questo
file (_HERE), cosi' il progetto e' portabile: non c'e' nessun /home/... o
/kaggle/input/... hardcodato. I dataset stanno in ./datasets/ accanto ai
moduli .py. Se sposti la cartella, i path continuano a funzionare.
"""
import os
import random
import numpy as np
import torch

# --- Ancora: cartella che contiene questo config.py ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "datasets")

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

# --- Parametri pipeline ---
NUM_KEYPOINTS = 17           # keypoint standard COCO
INPUT_SIZE    = (256, 192)   # (H, W) - standard pose estimation
HEATMAP_SIZE  = (64, 48)     # heatmap a 1/4 della risoluzione input
SIGMA         = 2            # ampiezza gaussiana nelle heatmap

# --- Path COCO 2017 Keypoints (LOCALE) ---
COCO_ROOT      = os.path.join(_DATA, "coco2017")
COCO_TRAIN_IMG = os.path.join(COCO_ROOT, "train2017")   # da scaricare per il fine-tuning
COCO_VAL_IMG   = os.path.join(COCO_ROOT, "val2017")
COCO_TRAIN_ANN = os.path.join(COCO_ROOT, "annotations", "person_keypoints_train2017.json")
COCO_VAL_ANN   = os.path.join(COCO_ROOT, "annotations", "person_keypoints_val2017.json")

# --- Path OCHuman (LOCALE, singolo annidamento) ---
OCHUMAN_IMG      = os.path.join(_DATA, "OCHuman", "images")
OCHUMAN_VAL_ANN  = os.path.join(_DATA, "OCHuman", "ochuman_coco_format_val_range_0.00_1.00.json")
OCHUMAN_TEST_ANN = os.path.join(_DATA, "OCHuman", "ochuman_coco_format_test_range_0.00_1.00.json")

# --- Output (checkpoint locali) ---
CHECKPOINT_DIR = os.path.join(_HERE, "checkpoints")
BEST_PTH       = os.path.join(_HERE, "models", "best.pth")   # baseline recuperata

# --- STL fine-tuning (valori condivisi tra celle 3c/3d/4b del notebook) ---
BONE_RATIO_THRESHOLD = 1.5    # soglia max/min per simmetria ossea (AVR + STL)
COLLAPSE_THRESHOLD   = 0.10   # segmento/torso minimo (AVR + STL)
MIN_CONF             = 0.3    # confidenza minima per l'AVR (score heatmap)
AVR_ANGLE_MIN_DEG    = 20.0   # floor angolare condiviso AVR e STL
STL_FINE_TUNE_LR     = 3e-5   # era 1e-4: ridotto per evitare catastrophic forgetting da E04
STL_TARGET_FRAC      = 0.1    # rho: frazione della spinta heatmap a cui portare ogni termine
STL_NUM_EPOCHS       = 10
STL_BETA             = 50     # soft_argmax piu' nitida, riduce gap train/eval