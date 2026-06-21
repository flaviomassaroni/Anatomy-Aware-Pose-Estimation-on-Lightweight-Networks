"""Globals: percorsi dataset, iperparametri, seed, device.
Sezione 'Globals' della struttura richiesta dalla prof.
"""
import random
import numpy as np
import torch

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

# --- Path COCO 2017 Keypoints (montato su Kaggle) ---
COCO_ROOT      = '/kaggle/input/datasets/asad11914/coco-2017-keypoints/coco2017'
COCO_TRAIN_IMG = f'{COCO_ROOT}/train2017'
COCO_VAL_IMG   = f'{COCO_ROOT}/val2017'
COCO_TRAIN_ANN = f'{COCO_ROOT}/annotations/person_keypoints_train2017.json'
COCO_VAL_ANN   = f'{COCO_ROOT}/annotations/person_keypoints_val2017.json'

# --- Path OCHuman (dataset Kaggle condiviso) ---
# ATTENZIONE: dopo aver montato il dataset nel runner, verifica lo slug reale
# con la cella os.walk. Se il path non corrisponde, correggi QUI e fai push.
OCHUMAN_IMG      = '/kaggle/input/ochuman/images/images'
OCHUMAN_VAL_ANN  = '/kaggle/input/ochuman/ochuman_coco_format_val_range_0.00_1.00.json'
OCHUMAN_TEST_ANN = '/kaggle/input/ochuman/ochuman_coco_format_test_range_0.00_1.00.json'

# --- Output (resta su Kaggle, NON su Git) ---
CHECKPOINT_DIR = '/kaggle/working/checkpoints'
