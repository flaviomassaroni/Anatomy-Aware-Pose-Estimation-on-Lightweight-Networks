"""Vincoli antropometrici per la Skeletal Topology Loss (STL).

Fonti biomeccaniche:
  [1] Winter, D.A. "Biomechanics and Motor Control of Human Movement",
      4th ed., Wiley, 2009. Chapter 3, Table 3.1, Figure 3.1.
  [2] Drillis, R. & Contini, R. "Body Segment Parameters", Technical
      Report No. 1166-03, NYU School of Engineering, 1966.
  [3] American Academy of Orthopaedic Surgeons (AAOS), "Joint Motion:
      Method of Measuring and Recording", 1965.

Note importanti:
  - I rapporti segmentali sono espressi come frazione dell'altezza H del
    soggetto. Poiche' in 2D monoculare non conosciamo H, usiamo i RAPPORTI
    TRA SEGMENTI (adimensionali), non le lunghezze assolute.
  - I range sono generosi (media ± ~3 SD o range fisiologico completo)
    per evitare di penalizzare pose inusuali ma valide.
  - In proiezione 2D gli angoli e le lunghezze appaiono accorciati
    (foreshortening). I range devono tenerne conto: usiamo margini
    conservativi, non range stretti.
"""

# ===================================================================
# COCO KEYPOINT INDICES (per riferimento)
# ===================================================================
# 0: nose        1: left_eye     2: right_eye    3: left_ear
# 4: right_ear   5: left_shoulder  6: right_shoulder
# 7: left_elbow  8: right_elbow  9: left_wrist  10: right_wrist
# 11: left_hip   12: right_hip   13: left_knee   14: right_knee
# 15: left_ankle  16: right_ankle

# ===================================================================
# 1. RAPPORTI DI LUNGHEZZA SEGMENTALE (bone ratio constraints)
# ===================================================================
#
# Da Winter 2009, Fig. 3.1 (originariamente Drillis & Contini 1966):
#
#   Segmento             Lunghezza / Altezza H
#   ─────────────────────────────────────────
#   Upper arm (spalla→gomito)       0.186 H
#   Forearm   (gomito→polso)        0.146 H
#   Thigh     (anca→ginocchio)      0.245 H
#   Shank     (ginocchio→caviglia)  0.246 H
#
# Da questi deriviamo i rapporti INTER-SEGMENTALI (adimensionali):
#
#   forearm / upper_arm  = 0.146 / 0.186 = 0.785
#   shank / thigh        = 0.246 / 0.245 = 1.004
#   upper_arm / thigh    = 0.186 / 0.245 = 0.759
#   forearm / shank      = 0.146 / 0.246 = 0.593
#
# I range qui sotto sono il valore nominale ± margine generoso (~30%)
# per assorbire variabilita' individuale + foreshortening 2D.

BONE_RATIOS = {
    # (segmento_numeratore, segmento_denominatore): (ratio_min, ratio_nominale, ratio_max)
    #
    # Ogni segmento e' definito come (keypoint_prossimale, keypoint_distale)

    # Avambraccio / Braccio superiore (stesso lato)
    'forearm_over_upper_arm': {
        'numerator':   (7, 9),    # left_elbow → left_wrist  (o 8,10 per dx)
        'denominator': (5, 7),    # left_shoulder → left_elbow (o 6,8 per dx)
        'nominal': 0.785,
        'range': (0.55, 1.05),    # ~30% margine su entrambi i lati
        'source': 'Winter 2009, 0.146H / 0.186H',
    },

    # Gamba inferiore / Coscia (stesso lato)
    'shank_over_thigh': {
        'numerator':   (13, 15),  # left_knee → left_ankle  (o 14,16 per dx)
        'denominator': (11, 13),  # left_hip → left_knee (o 12,14 per dx)
        'nominal': 1.004,
        'range': (0.70, 1.35),
        'source': 'Winter 2009, 0.246H / 0.245H',
    },

    # Braccio / Coscia (cross-limb, utile per coerenza globale)
    'upper_arm_over_thigh': {
        'numerator':   (5, 7),    # left_shoulder → left_elbow
        'denominator': (11, 13),  # left_hip → left_knee
        'nominal': 0.759,
        'range': (0.50, 1.05),
        'source': 'Winter 2009, 0.186H / 0.245H',
    },
}

# Coppie simmetriche: il rapporto lunghezza_sx / lunghezza_dx deve
# essere vicino a 1.0. Margine: [0.65, 1.55] (~55% asimmetria max).
# Nella baseline AVR usiamo 1.5; qui e' identico.
SYMMETRY_PAIRS = [
    # (kp_pair_left, kp_pair_right, nome)
    ((5, 7),   (6, 8),   'upper_arm'),     # spalla→gomito
    ((7, 9),   (8, 10),  'forearm'),        # gomito→polso
    ((11, 13), (12, 14), 'thigh'),          # anca→ginocchio
    ((13, 15), (14, 16), 'shank'),          # ginocchio→caviglia
]
SYMMETRY_RANGE = (0.65, 1.55)


# ===================================================================
# 2. RANGE ANGOLARI ARTICOLARI (joint angle constraints)
# ===================================================================
#
# Fonti: AAOS 1965, Winter 2009 Chapter 2.
#
# ATTENZIONE: questi sono i range dell'angolo INCLUSO al vertice del
# giunto, calcolato in 2D come angolo tra i due vettori adiacenti.
# NON sono gli angoli goniometrici clinici (che misurano flessione/
# estensione da una posizione anatomica neutra).
#
# Angolo 2D al giunto = angolo tra vettore(giunto→prossimale) e
#                        vettore(giunto→distale)
#
# Esempi:
#   - Gomito completamente esteso → angolo ≈ 180°
#   - Gomito completamente flesso → angolo ≈ 30-40°
#   - Ginocchio esteso → angolo ≈ 175-180° (lieve iperestensione)
#   - Ginocchio completamente flesso → angolo ≈ 30-40°
#
# In proiezione 2D gli angoli possono apparire piu' chiusi del reale
# (foreshortening), quindi il minimo e' piu' basso di quello clinico.

JOINT_ANGLE_RANGES = {
    # (kp_prossimale, kp_giunto, kp_distale): (angolo_min_deg, angolo_max_deg)

    # Gomito sinistro: spalla → gomito → polso
    'left_elbow':  {'joints': (5, 7, 9),   'range_deg': (10.0, 180.0)},

    # Gomito destro
    'right_elbow': {'joints': (6, 8, 10),  'range_deg': (10.0, 180.0)},

    # Ginocchio sinistro: anca → ginocchio → caviglia
    'left_knee':   {'joints': (11, 13, 15), 'range_deg': (10.0, 180.0)},

    # Ginocchio destro
    'right_knee':  {'joints': (12, 14, 16), 'range_deg': (10.0, 180.0)},

    # Spalla sinistra: anca_sx → spalla_sx → gomito_sx
    # La spalla ha ROM molto ampio; vincoli piu' laschi
    'left_shoulder':  {'joints': (11, 5, 7),  'range_deg': (10.0, 180.0)},

    # Spalla destra
    'right_shoulder': {'joints': (12, 6, 8),  'range_deg': (10.0, 180.0)},

    # Anca sinistra: spalla_sx → anca_sx → ginocchio_sx
    'left_hip':    {'joints': (5, 11, 13), 'range_deg': (15.0, 180.0)},

    # Anca destra
    'right_hip':   {'joints': (6, 12, 14), 'range_deg': (15.0, 180.0)},
}

# Nota per il paper: i range sono volutamente generosi (10° minimo
# invece dei 30-40° clinici) per assorbire il foreshortening 2D.
# Il punto NON e' replicare i range clinici esatti, ma escludere le
# predizioni palesemente impossibili (angolo ~0° = collasso, angolo
# >180° = giunto invertito).


# ===================================================================
# 3. VINCOLI DI ORDINAMENTO GEOMETRICO (geometric ordering)
# ===================================================================
#
# Il corpo umano ha una struttura ad albero cinematico. Lungo ogni
# catena, i giunti intermedi devono stare GEOMETRICAMENTE TRA gli
# estremi (proiettati sulla direzione della catena).
#
# In pratica: il ginocchio deve stare "tra" anca e caviglia lungo
# l'asse della gamba, non sopra l'anca o sotto la caviglia. In 2D
# questo si traduce in un vincolo sulla proiezione.
#
# Implementazione: per ogni terna (a, mid, b), calcoliamo:
#   t = dot(mid - a, b - a) / dot(b - a, b - a)
# Se t e' in [0, 1], mid sta tra a e b (proiettato). Se t < 0 o t > 1,
# mid e' fuori dal segmento → violazione.
#
# Penalita' differenziabile:
#   L_order = max(0, -t)^2 + max(0, t - 1)^2

KINEMATIC_CHAINS = [
    # (estremo_prossimale, giunto_intermedio, estremo_distale, nome)
    (5,  7,  9,  'left_arm'),        # spalla → gomito → polso
    (6,  8,  10, 'right_arm'),
    (11, 13, 15, 'left_leg'),        # anca → ginocchio → caviglia
    (12, 14, 16, 'right_leg'),
]


# ===================================================================
# 4. TABELLA RIASSUNTIVA PER IL PAPER
# ===================================================================
#
# Constraint Type     | Source              | #Rules | What it catches
# ────────────────────┼─────────────────────┼────────┼──────────────────
# Bone ratio          | Winter 2009,        |   3    | Segmenti troppo
#   (inter-segment)   | Drillis&Contini 1966|        | lunghi/corti
# ────────────────────┼─────────────────────┼────────┼──────────────────
# Bone symmetry       | Anatomia generale   |   4    | Asimmetria sx/dx
#   (left vs right)   |                     |        | implausibile
# ────────────────────┼─────────────────────┼────────┼──────────────────
# Joint angle         | AAOS 1965,          |   8    | Giunti collassati
#                     | Winter 2009 Ch.2    |        | o invertiti
# ────────────────────┼─────────────────────┼────────┼──────────────────
# Geometric ordering  | Kinematic tree      |   4    | Giunto fuori
#                     |                     |        | dalla catena
# ────────────────────┼─────────────────────┼────────┼──────────────────
# TOTALE              |                     |  19    |
#
# Differenza chiave da Han et al. (2025):
#   Han → apprende medie e deviazioni standard DAL DATASET (vincoli
#          statistici, dataset-specific, non generalizzano).
#   Noi  → range da letteratura biomeccanica INDIPENDENTE dal dataset
#          (vincoli assoluti, generalizzano cross-dataset, citabili).
