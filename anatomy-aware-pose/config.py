"""Globals: percorsi dataset, iperparametri, seed, device. Sezione 'Globals' della struttura richiesta dalla prof.
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
OCHUMAN_IMG      = '/kaggle/input/datasets/messinaalberto/ochuman/images/images'
OCHUMAN_VAL_ANN  = '/kaggle/input/datasets/messinaalberto/ochuman/ochuman_coco_format_val_range_0.00_1.00.json'
OCHUMAN_TEST_ANN = '/kaggle/input/datasets/messinaalberto/ochuman/ochuman_coco_format_test_range_0.00_1.00.json'

# --- Output (resta su Kaggle, NON su Git) ---
CHECKPOINT_DIR = '/kaggle/working/checkpoints'

# --- STL fine-tuning (valori wirrati nelle celle 3c e 4b del notebook) ---
BONE_RATIO_THRESHOLD = 1.5    # soglia max/min per simmetria ossea (AVR + STL)
COLLAPSE_THRESHOLD   = 0.10   # segmento/torso minimo (AVR + STL)
MIN_CONF             = 0.3    # confidenza minima per l'AVR (score heatmap)
AVR_ANGLE_MIN_DEG    = 20.0   # floor angolare condiviso AVR e STL
STL_FINE_TUNE_LR     = 3e-5   # era 1e-4: ridotto per evitare catastrophic forgetting da E04
STL_TARGET_FRAC      = 0.1    # era 1.0: STL pesava 3:1 su heatmap (causa gaming)
STL_NUM_EPOCHS       = 10
STL_BETA             = 50     # beta=50: mediana gap 0.27 px, p95 3.2 (vs beta=30 p95=9.3); vedi analisi decoder gap cella 3d
