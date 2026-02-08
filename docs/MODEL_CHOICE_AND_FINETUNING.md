# Model choice: fine-tuning vs small model (≤2 MB)

## Does fine-tuning increase accuracy and precision?

**Yes, usually** — with a large pretrained model (e.g. ResNet50), **fine-tuning** (Phase 2: unfreezing last layers and training with a low learning rate) typically **increases** test accuracy and precision compared to **classifier-only** (Phase 1: frozen backbone).

- **Why:** The backbone adapts to your defect images; the last layers learn task-specific features.
- **Risk:** On **small datasets** (e.g. 875 images), fine-tuning can **overfit**: validation/test accuracy looks high but the model memorizes. Your earlier ResNet run reached 100% val/test, which was likely overfitting.
- **Recommendation:** If you need **high accuracy** and have no strict size limit, use ResNet with **limited** fine-tuning (last 2 layers only) and **strong regularization** (dropout, weight decay, label smoothing, early stopping). If you need **≤2 MB**, use the small model and **no** fine-tuning (single-phase training only).

## 2 MB max: use the small model

To meet a **2 MB model size** and **avoid overfitting**:

1. **Use `--model small`** (default in `train.py`). This trains **SmallDefectNet** (~1.62 MB, ~423K parameters).
2. **Single-phase training only** — no “fine-tuning” in the ResNet sense. The whole network is trained from scratch with:
   - **Regularization:** dropout (0.5, 0.4 in classifier), weight decay 1e-4, label smoothing 0.1
   - **Early stopping:** patience 10 (stops if validation accuracy does not improve)
   - **Moderate capacity:** small CNN (4 conv blocks + 2 FC) so it cannot easily memorize

3. **Expected:** Lower accuracy than ResNet (e.g. ~50–65% test vs 82% with ResNet), but **no overfitting** (train and val accuracy stay in a similar range) and **model size under 2 MB**.

### Commands

```bash
# Train small model (≤2 MB, single phase)
python train.py --model small --data_dir data/raw --epochs 50 --patience 10

# Export to ONNX (~1.7 MB total)
python export_onnx.py --model small
# -> models/defect_8class_small.onnx (+ .onnx.data)
```

### Sizes

| Artifact              | Size    |
|-----------------------|---------|
| SmallDefectNet (params) | ~1.62 MB |
| defect_8class_small.onnx + .data | ~1.7 MB |
| ResNet50 best_model.pth | ~96 MB  |

## Summary

| Goal              | Use              | Fine-tuning? | Size   | Overfitting risk |
|-------------------|------------------|---------------|--------|-------------------|
| Max accuracy      | ResNet50         | Limited (last 2 layers) | ~96 MB | Medium (use regularization) |
| **≤2 MB, no overfit** | **SmallDefectNet** | **No (single phase)** | **~1.6 MB** | **Low** |
