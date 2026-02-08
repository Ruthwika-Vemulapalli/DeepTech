# Phase 1 Slide Content — DeepTech Hackathon 2026

**File name:** `TeamName_Phase1.pdf`  
**Max slides:** 6–7 including title

---

## Slide 1: Title + Team + PS code + Phase tag
- Title: Semiconductor Defect Classification (8-Class)
- Team name: [Your Team Name]
- Problem Statement code: [PS code]
- Phase: Phase 1

---

## Slide 2: Problem statement
- Your understanding of the problem (not copy-paste): e.g. Classify semiconductor manufacturing defects from SEM/wafer images into at least 6 defect types plus Clean and Other for quality control.

---

## Slide 3: Idea summary + approach
- Idea: Transfer learning with ResNet50 for 8-class defect classification (6 defect + Clean + Other).
- Approach: Two-phase training (classifier head then limited backbone fine-tuning), same as prior binary model.

---

## Slide 4: Proposed solution (architecture + Dataset plan & class design)

### Architecture overview
- ResNet50 backbone + custom classifier head; two-phase training.

### Dataset Plan & Class Design (must-have)
- **Total images planned/current:** 875
- **No. of classes:** 8 (Min 8: 6 defect + Clean + Other)
- **Class list:** Defect_1 (Bridge), Defect_2 (CMP scratch), Defect_3 (LER), Defect_4 (Opens), Defect_5 (Crack), Defect_6 (Short), Clean, Other
- **Class balance plan:** min 50 per class (Clean 375, defect types 75 each, Other 50)
- **Train/Val/Test split:** 70% / 15% / 15%
- **Image type:** Grayscale preferred (current images 224×224 RGB; grayscale can be used in preprocessing)
- **Labeling method/source:** manual (SEM defect images); reference public dataset: MIR-WM811K

---

## Slide 5: Technology & Feasibility / Methodology — Baseline model & results (Phase 1)

### Model details
- **Architecture:** ResNet50
- **Training approach:** transfer learning
- **Input size:** 224 × 224
- **Model size:** ~96 MB (PyTorch); ONNX ~96 MB
- **Framework:** PyTorch

### Metrics on test split
- **Accuracy:** 82.44%
- **Precision:** 79.76%
- **Recall:** 82.44%
- **F1 (weighted):** 79.82%
- **Confusion Matrix:** (insert image: `results/confusion_matrix.png` or table from `results/confusion_matrix.csv`)

---

## Slide 6: Artifacts & Links

### Mandatory links
- **GitHub Repository (mandatory):** https://github.com/Ruthwika-Vemulapalli/DeepTech
- **Dataset ZIP link (Drive/Kaggle/HF):** [Upload data/raw or data ZIP and add link]
- **ONNX model link:** [Upload `models/defect_8class.onnx` + `defect_8class.onnx.data` or a ZIP and add link]
- **Results report link (optional):** [Link to this repo’s results/ or a PDF report]

### Video
- Optional for Phase 1 (not mandatory).

---

## Slide 7: Research & References (keep light)
- 2–3 references max.
- Dataset source: MIR-WM811K — Ming-Ju Wu, Jyh-Shing Roger Jang, Jui-Long Chen, “Wafer Map Failure Pattern Recognition and Similarity Ranking for Large-Scale Data Sets,” IEEE Trans. Semiconductor Manufacturing, vol. 28, no. 1, 2015. http://mirlab.org/dataset/public/
- [Add 1–2 more: e.g. deep learning defect classification paper; NXP eIQ docs if used]
