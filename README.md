# DeepTech — Semiconductor Defect Classification (Phase 1)

8-class defect classification for semiconductor manufacturing: **6 defect types + Clean + Other**, using ResNet50 and transfer learning (same two-phase training as the earlier binary model).

**GitHub:** https://github.com/Ruthwika-Vemulapalli/DeepTech

## Classes (8)

| Index | Class        | Description        |
|-------|--------------|--------------------|
| 0     | bridge       | Defect_1           |
| 1     | cmp_scratch  | Defect_2           |
| 2     | ler          | Defect_3           |
| 3     | opens        | Defect_4           |
| 4     | crack        | Defect_5 (via_crack) |
| 5     | short        | Defect_6           |
| 6     | clean        | Clean              |
| 7     | other        | Other              |

## Setup

```bash
pip install -r requirements.txt
# Optional for ONNX export: pip install onnx onnxscript
```

## Data (8-class)

Prepare data under `data/raw/` with one subfolder per class: `bridge`, `cmp_scratch`, `ler`, `opens`, `crack`, `short`, `clean`, `other`, each containing PNG (or JPG) images.

From source folders (e.g. bridge_clean_150_highquality, via_crack_clean_defect_150_highquality, …):

```bash
python scripts/prepare_8class_data.py --source_dir /path/to/defect/folder --out_dir data/raw
```

## Train

**Default: small model (≤2 MB, single phase, strong regularization to avoid overfitting)**

```bash
python train.py --model small --data_dir data/raw
```

**ResNet50 (two-phase, higher accuracy, ~96 MB):**

```bash
python train.py --model resnet --data_dir data/raw
```

Options: `--model small|resnet`, `--epochs`, `--patience`, `--batch_size`, `--lr`. For ResNet: `--epochs_finetune`, `--lr_finetune`.

## Export ONNX

```bash
# Small model (~1.7 MB, for ≤2 MB constraint)
python export_onnx.py --model small
# -> models/defect_8class_small.onnx

# ResNet50 (~96 MB)
python export_onnx.py --model resnet
# -> models/defect_8class.onnx
```

## Results (Phase 1)

**ResNet50 (82% test accuracy, ~96 MB):**
- Test accuracy: 82.44%; Precision / Recall: 79.76% / 82.44%
- Confusion matrix: `results/confusion_matrix.png`

**Small model (≤2 MB, no overfitting):**
- Model size: **1.62 MB** (fits 2 MB limit)
- Single-phase training; accuracy typically ~50–65% (lower than ResNet but stable)
- Export: `python export_onnx.py --model small` → `models/defect_8class_small.onnx` (~1.7 MB)

## Project layout

```
DeepTech/
├── data/raw/           # 8-class images (bridge, cmp_scratch, ler, opens, crack, short, clean, other)
├── models/             # best_model.pth, defect_8class.onnx
├── results/            # confusion_matrix.png, confusion_matrix.csv
├── src/
│   ├── data/           # dataset, loader, augmentation
│   ├── models/         # ResNet50 classifier
│   ├── training/       # trainer
│   └── utils/          # metrics
├── scripts/            # prepare_8class_data.py, save_confusion_matrix.py
├── train.py
├── export_onnx.py
├── requirements.txt
├── PHASE1_SLIDE_CONTENT.md
└── README.md
```

## Phase 1 deliverables

- **Dataset plan & class design:** See `PHASE1_SLIDE_CONTENT.md`
- **Baseline model & results:** ResNet50, 224×224, PyTorch; metrics and confusion matrix above. 
- **Artifacts & links:** GitHub (mandatory), Dataset ZIP, ONNX link, optional results report. 
- **Research & references:** 2–3 refs + dataset citation. 

