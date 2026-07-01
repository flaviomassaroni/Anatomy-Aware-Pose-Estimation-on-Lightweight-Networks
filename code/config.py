"""Globals: percorsi dataset, iperparametri, seed, device. Sezione 'Globals'.
Auto-detect LOCALE vs KAGGLE: un solo file su Git, path scelti a runtime.
Override manuale: POSE_ENV=local o POSE_ENV=kaggle.
"""
import os
import random
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.environ.get("POSE_DATA_DIR", os.path.join(_HERE, "..", "datasets"))

_forced = os.environ.get("POSE_ENV", "").strip().lower()
if _forced in ("local", "kaggle"):
    IS_LOCAL = (_forced == "local")
elif os.path.isdir(_DATA):
    IS_LOCAL = True
elif os.path.isdir("/kaggle/input"):
    IS_LOCAL = False
else:
    IS_LOCAL = False  

ENV_NAME = "LOCAL" if IS_LOCAL else "KAGGLE"

SEED = 42

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_KEYPOINTS = 17           # keypoint standard COCO
INPUT_SIZE    = (256, 192)   # (H, W) - standard pose estimation
HEATMAP_SIZE  = (64, 48)     # heatmap a 1/4 della risoluzione input
SIGMA         = 2            # ampiezza gaussiana nelle heatmap

if IS_LOCAL:
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
    RESULTS_DIR    = os.path.join(_HERE, "..", "results")   # root results/, non code/results/
else:
    COCO_ROOT      = '/kaggle/input/datasets/asad11914/coco-2017-keypoints/coco2017'
    COCO_TRAIN_IMG = f'{COCO_ROOT}/train2017'
    COCO_VAL_IMG   = f'{COCO_ROOT}/val2017'
    COCO_TRAIN_ANN = f'{COCO_ROOT}/annotations/person_keypoints_train2017.json'
    COCO_VAL_ANN   = f'{COCO_ROOT}/annotations/person_keypoints_val2017.json'

    OCHUMAN_IMG      = '/kaggle/input/datasets/messinaalberto/ochuman/images/images'
    OCHUMAN_VAL_ANN  = '/kaggle/input/datasets/messinaalberto/ochuman/ochuman_coco_format_val_range_0.00_1.00.json'
    OCHUMAN_TEST_ANN = '/kaggle/input/datasets/messinaalberto/ochuman/ochuman_coco_format_test_range_0.00_1.00.json'

    CHECKPOINT_DIR = '/kaggle/working/checkpoints'
    BEST_PTH       = '/kaggle/input/datasets/messinaalberto/pose-baseline-checkpoint/best.pth'
    RESULTS_DIR    = '/kaggle/working'

BONE_RATIO_THRESHOLD = 1.5    # soglia max/min per simmetria ossea (AVR + STL)
COLLAPSE_THRESHOLD   = 0.10   # segmento/torso minimo (AVR + STL)
MIN_CONF             = 0.3    # confidenza minima per l'AVR (score heatmap)
AVR_ANGLE_MIN_DEG    = 20.0   # floor angolare condiviso AVR e STL
STL_FINE_TUNE_LR     = 1e-5   # LR basso: evita catastrophic forgetting sul backbone
STL_TARGET_FRAC      = 0.1    # rho: frazione della spinta heatmap a cui portare ogni termine
STL_BETA             = 50     # temperatura soft-argmax: gap mediano 0.27 px, p95 3.2 px con sigma=2
STL_BONE_BOOST       = 5.0    # moltiplicatore bone loss nel run sev_E7
STL_SEV_EPOCHS       = 8      # numero epoche run sev (checkpoint selezionato a E7)
SEVERITY_ALPHA       = 2.0    # peso severity weighting (0.0 = hinge pura); vedi stl.py