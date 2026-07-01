"""Vincoli antropometrici per la STL: rapporti segmentali, simmetria, angoli, ordering.
Fonti: [1] Winter 2009, Cap.3 Tab.3.1. [2] Drillis & Contini 1966. [3] AAOS 1965.
"""
from config import AVR_ANGLE_MIN_DEG


# Indici COCO: 0=nose, 1=l_eye, 2=r_eye, 3=l_ear, 4=r_ear,
# 5=l_shoulder, 6=r_shoulder, 7=l_elbow, 8=r_elbow, 9=l_wrist, 10=r_wrist,
# 11=l_hip, 12=r_hip, 13=l_knee, 14=r_knee, 15=l_ankle, 16=r_ankle

# Rapporti inter-segmentali adimensionali da Winter 2009 (forearm/upper_arm, shank/thigh, ...).
# Range ~±30%: assorbe variabilita' individuale + foreshortening 2D.
BONE_RATIOS = {
    'forearm_over_upper_arm': {
        'numerator':   (7, 9),    # left_elbow → left_wrist
        'denominator': (5, 7),    # left_shoulder → left_elbow
        'nominal': 0.785,
        'range': (0.55, 1.05),
        'source': 'Winter 2009, 0.146H / 0.186H',
    },
    'shank_over_thigh': {
        'numerator':   (13, 15),  # left_knee → left_ankle
        'denominator': (11, 13),  # left_hip → left_knee
        'nominal': 1.004,
        'range': (0.70, 1.35),
        'source': 'Winter 2009, 0.246H / 0.245H',
    },
    'upper_arm_over_thigh': {
        'numerator':   (5, 7),    # left_shoulder → left_elbow
        'denominator': (11, 13),  # left_hip → left_knee
        'nominal': 0.759,
        'range': (0.50, 1.05),
        'source': 'Winter 2009, 0.186H / 0.245H',
    },
}

# Simmetria sx/dx: ratio lunghezza_sx/lunghezza_dx atteso vicino a 1.0.
# Soglia coincide con BONE_RATIO_THRESHOLD (1.5) dell'AVR.
SYMMETRY_PAIRS = [
    ((5, 7),   (6, 8),   'upper_arm'),
    ((7, 9),   (8, 10),  'forearm'),
    ((11, 13), (12, 14), 'thigh'),
    ((13, 15), (14, 16), 'shank'),
]

# Angolo incluso al vertice del giunto (non angolo clinico goniometrico).
# Floor AVR_ANGLE_MIN_DEG=20° (< clinico) per assorbire foreshortening 2D.
# Solo gomiti e ginocchia: allineati ai KPI AVR. Spalla/anca rimossi (ROM quasi 180°, segnale nullo).
JOINT_ANGLE_RANGES = {
    'left_elbow':  {'joints': (5, 7, 9),    'range_deg': (AVR_ANGLE_MIN_DEG, 180.0)},
    'right_elbow': {'joints': (6, 8, 10),   'range_deg': (AVR_ANGLE_MIN_DEG, 180.0)},
    'left_knee':   {'joints': (11, 13, 15), 'range_deg': (AVR_ANGLE_MIN_DEG, 180.0)},
    'right_knee':  {'joints': (12, 14, 16), 'range_deg': (AVR_ANGLE_MIN_DEG, 180.0)},
}

# Giunti intermedi vincolati a stare tra gli estremi della catena (proiettato).
# geometric_ordering_loss non e' un KPI AVR; lambda_order calibrato separatamente.
KINEMATIC_CHAINS = [
    (5,  7,  9,  'left_arm'),
    (6,  8,  10, 'right_arm'),
    (11, 13, 15, 'left_leg'),
    (12, 14, 16, 'right_leg'),
]
