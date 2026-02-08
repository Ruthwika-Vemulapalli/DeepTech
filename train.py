"""
Train 8-class defect model.
Use --model small for <=2 MB model (single phase, strong regularization, no overfitting).
Use --model resnet for two-phase ResNet50 (larger, transfer learning).
"""
import torch
import argparse
from pathlib import Path
from src.data.loader import create_dataloaders
from src.data.dataset import NUM_CLASSES, CLASS_NAMES
from src.models.resnet import ResNet50Classifier
from src.models.small_cnn import SmallDefectNet, model_size_mb
from src.training.trainer import Trainer
from src.utils.metrics import evaluate_model


def main():
    p = argparse.ArgumentParser(description='Train 8-class defect model')
    p.add_argument('--model', type=str, default='small', choices=['small', 'resnet'],
                   help='small: <=2 MB, single phase. resnet: two-phase transfer learning')
    p.add_argument('--data_dir', type=str, default='data/raw')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--epochs', type=int, default=50, help='Total epochs (small: single phase; resnet: classifier only)')
    p.add_argument('--epochs_finetune', type=int, default=30, help='ResNet only: fine-tune epochs')
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--lr_finetune', type=float, default=0.0001)
    p.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--save_name', type=str, default=None, help='Checkpoint name (default: best_model.pth or best_small_model.pth)')
    args = p.parse_args()

    use_small = args.model == 'small'
    save_name = args.save_name or ('best_small_model.pth' if use_small else 'best_model.pth')
    save_path = Path('models') / save_name
    Path('models').mkdir(exist_ok=True)

    print('=' * 60)
    print('Semiconductor Defect Detection - 8-Class Training')
    print('=' * 60)
    print(f'Model: {args.model}  Device: {args.device}  Batch size: {args.batch_size}')
    print(f'Classes: {NUM_CLASSES} - {CLASS_NAMES}')
    print()

    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=4 if args.device == 'cuda' else 0
    )
    print(f'Train: {len(train_loader.dataset)}  Val: {len(val_loader.dataset)}  Test: {len(test_loader.dataset)}')
    print()

    if use_small:
        model = SmallDefectNet(num_classes=NUM_CLASSES)
        size_mb = model_size_mb(model)
        print(f'SmallDefectNet size: {size_mb:.2f} MB (max 2 MB)')
        assert size_mb <= 2.1, f'Model {size_mb:.2f} MB exceeds 2 MB'
        trainer = Trainer(model, args.device, save_dir='models', save_best_as=save_name)
        # Single phase: full model, strong regularization (dropout in model + weight_decay + label_smoothing)
        print('Training small model (single phase, strong regularization to avoid overfitting)')
        trainer.train(
            train_loader, val_loader,
            epochs=args.epochs,
            lr=args.lr,
            phase='fine_tune',  # train all params
            patience=args.patience
        )
    else:
        model = ResNet50Classifier(num_classes=NUM_CLASSES, pretrained=True, freeze_backbone=True)
        trainer = Trainer(model, args.device, save_dir='models', save_best_as=save_name)
        print('PHASE 1: Training classifier head')
        trainer.train(
            train_loader, val_loader,
            epochs=args.epochs,
            lr=args.lr,
            phase='classifier',
            patience=args.patience
        )
        print('PHASE 2: Limited fine-tuning (last 2 layers)')
        model.unfreeze_partial_backbone(num_layers=2)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f'Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)')
        trainer.train(
            train_loader, val_loader,
            epochs=args.epochs_finetune,
            lr=args.lr_finetune,
            phase='fine_tune',
            patience=7
        )

    print('Evaluating on test set')
    ckpt = torch.load(str(save_path), map_location=args.device)
    model.load_state_dict(ckpt['model_state_dict'])
    results = evaluate_model(model, test_loader, args.device)

    print('\nTest Results:')
    print(f'Accuracy:  {results["accuracy"]*100:.2f}%')
    print(f'Precision: {results["precision"]*100:.2f}%')
    print(f'Recall:    {results["recall"]*100:.2f}%')
    print(f'F1:       {results["f1"]*100:.2f}%')
    if use_small:
        print(f'Model size: {model_size_mb(model):.2f} MB')
    print('Confusion Matrix:')
    print(results['confusion_matrix'])
    print('Training completed.')


if __name__ == '__main__':
    main()
