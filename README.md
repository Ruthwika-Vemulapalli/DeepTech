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

Same procedure as before: Phase 1 = classifier only, Phase 2 = limited fine-tuning (last 2 layers).

```bash
python train.py --data_dir data/raw
```

Options: `--batch_size`, `--epochs_classifier`, `--epochs_finetune`, `--lr_classifier`, `--lr_finetune`.

## Export ONNX

```bash
python export_onnx.py
# Output: models/defect_8class.onnx (+ .onnx.data)
```

## Results (Phase 1)

- **Test accuracy:** 82.44%
- **Precision / Recall (weighted):** 79.76% / 82.44%
- **Confusion matrix:** `results/confusion_matrix.png`, `results/confusion_matrix.csv`

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

- **Dataset plan & class design:** See `PHASE1_SLIDE_CONTENT.md` (slide 4).
- **Baseline model & results:** ResNet50, 224×224, PyTorch; metrics and confusion matrix above (slide 5).
- **Artifacts & links:** GitHub (mandatory), Dataset ZIP, ONNX link, optional results report (slide 6).
- **Research & references:** 2–3 refs + dataset citation (slide 7).

Save slides as **TeamName_Phase1.pdf** (6–7 slides max).
