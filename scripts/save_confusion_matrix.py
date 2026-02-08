"""
Load best model, run on test set, save confusion matrix as image and CSV for slides
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from src.data.loader import create_dataloaders
from src.data.dataset import NUM_CLASSES, CLASS_NAMES
from src.models.resnet import ResNet50Classifier
from src.utils.metrics import evaluate_model


def main():
    data_dir = 'data/raw'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _, _, test_loader = create_dataloaders(data_dir=data_dir, batch_size=32, num_workers=0)
    model = ResNet50Classifier(num_classes=NUM_CLASSES, pretrained=False)
    ckpt = torch.load('models/best_model.pth', map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    results = evaluate_model(model, test_loader, device)
    cm = results['confusion_matrix']

    Path('results').mkdir(exist_ok=True)
    np.savetxt('results/confusion_matrix.csv', cm, delimiter=',', fmt='%d')

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(np.arange(NUM_CLASSES))
    ax.set_yticks(np.arange(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    ax.set_yticklabels(CLASS_NAMES)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, int(cm[i, j]), ha='center', va='center', color='black')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix (Phase 1 - 8 Class)')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png', dpi=150)
    plt.close()
    print('Saved results/confusion_matrix.png and results/confusion_matrix.csv')


if __name__ == '__main__':
    main()
