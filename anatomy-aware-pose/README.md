# Anatomy-Aware Pose Estimation on Lightweight Networks

Human pose estimation anatomicamente vincolata (Skeletal Topology Loss) su backbone
leggero MobileNetV3, per deployment edge in scenari HRI.
Corso di Computer Vision — Prof. Irene Amerini, Sapienza.

## Struttura del codice
Segue lo schema richiesto (Imports → Globals → Utils → Data → Network → Train → Evaluation):

| File | Sezione | Contenuto |
|------|---------|-----------|
| `config.py` | Globals | path dataset, iperparametri, seed, device |
| `utils.py` | Utils | `generate_heatmap`, `decode_heatmaps`, `heatmap_to_original`, `count_params` |
| `data.py` | Data | `build_samples`, `COCOKeypointsDataset`, `COCOEvalDataset` |
| `network.py` | Network | `DeconvHead`, `PoseMobileNet` (MobileNetV3-Small) |
| `train.py` | Train | `WeightedMSELoss`, `train_one_epoch`, `validate`, `fit` |
| `evaluation.py` | Evaluation | inferenza, AP/AR (pycocotools), AVR, `evaluate_on_coco_val`, `evaluate_on_ochuman` |
| `kaggle_runner.ipynb` | — | notebook minimale che gira su Kaggle |

## Workflow (metodo a prova di bomba)
- **GitHub = unica fonte di verita'** per il codice. Si edita in locale (VSCode), si fa push.
- **Kaggle = solo macchina di training/eval.** Il runner clona il repo e importa i moduli.
- I **dataset** stanno su Kaggle (mai su Git). I **checkpoint** (`best.pth`) restano in
  `/kaggle/working` (mai su Git, sono pesanti).

### Per girare su Kaggle
1. Carica `kaggle_runner.ipynb` (o copia le sue celle in un nuovo notebook).
2. Settings → **Internet: On** (serve al `git clone`).
3. **Add Input**: aggiungi *COCO 2017 Keypoints* e il dataset *OCHuman* condiviso.
4. Nella cella 1 metti l'URL del repo. Repo privato → token via Add-ons → Secrets.
5. Run All. Il training salva `best.pth`; la cella di eval stampa AP/AR/AVR su COCO e OCHuman.

### Per lavorare in due
- Si edita in locale, `git pull` prima di iniziare, `git push` quando si finisce un pezzo.
- Su Kaggle il runner ri-clona ad ogni esecuzione: prende sempre l'ultima versione.

## Riproducibilita'
Seed fissi (`SEED=42`) e `cudnn.deterministic=True` in `config.set_seed()`, chiamato dal runner.
