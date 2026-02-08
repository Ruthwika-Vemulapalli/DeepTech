"""
Train 8-class defect model (same two-phase ResNet50 procedure as before)
"""
import torch
import argparse
from src.data.loader import create_dataloaders
from src.data.dataset import NUM_CLASSES, CLASS_NAMES
from src.models.resnet import ResNet50Classifier
from src.training.trainer import Trainer
from src.utils.metrics import evaluate_model


def main():
    p = argparse.ArgumentParser(description='Train 8-class defect model')
    p.add_argument('--data_dir', type=str, default='data/raw')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--epochs_classifier', type=int, default=15)
    p.add_argument('--epochs_finetune', type=int, default=30)
    p.add_argument('--lr_classifier', type=float, default=0.001)
    p.add_argument('--lr_finetune', type=float, default=0.0001)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    print('=' * 60)
    print('Semiconductor Defect Detection - 8-Class Training (Phase 1)')
    print('=' * 60)
    print(f'Device: {args.device}  Batch size: {args.batch_size}')
    print(f'Classes: {NUM_CLASSES} - {CLASS_NAMES}')
    print()

    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=4 if args.device == 'cuda' else 0
    )
    print(f'Train: {len(train_loader.dataset)}  Val: {len(val_loader.dataset)}  Test: {len(test_loader.dataset)}')
    print()

    model = ResNet50Classifier(num_classes=NUM_CLASSES, pretrained=True, freeze_backbone=True)
    trainer = Trainer(model, args.device)

    print('PHASE 1: Training classifier head')
    trainer.train(
        train_loader, val_loader,
        epochs=args.epochs_classifier,
        lr=args.lr_classifier,
        phase='classifier'
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
    ckpt = torch.load('models/best_model.pth', map_location=args.device)
    model.load_state_dict(ckpt['model_state_dict'])
    results = evaluate_model(model, test_loader, args.device)

    print('\nTest Results:')
    print(f'Accuracy:  {results["accuracy"]*100:.2f}%')
    print(f'Precision: {results["precision"]*100:.2f}%')
    print(f'Recall:    {results["recall"]*100:.2f}%')
    print(f'F1:       {results["f1"]*100:.2f}%')
    print('Confusion Matrix:')
    print(results['confusion_matrix'])
    print('Training completed.')


if __name__ == '__main__':
    main()
