"""
Export PyTorch model to ONNX. Use --model small for <=2 MB model.
"""
import torch
import argparse
from pathlib import Path
from src.models.resnet import ResNet50Classifier
from src.models.small_cnn import SmallDefectNet
from src.data.dataset import NUM_CLASSES


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, default='small', choices=['small', 'resnet'])
    p.add_argument('--checkpoint', type=str, default=None, help='Path to .pth (default: best_small_model.pth or best_model.pth)')
    p.add_argument('--output', type=str, default=None, help='Output ONNX path')
    args = p.parse_args()

    use_small = args.model == 'small'
    ckpt_path = args.checkpoint or ('models/best_small_model.pth' if use_small else 'models/best_model.pth')
    out_path = args.output or ('models/defect_8class_small.onnx' if use_small else 'models/defect_8class.onnx')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if use_small:
        model = SmallDefectNet(num_classes=NUM_CLASSES)
    else:
        model = ResNet50Classifier(num_classes=NUM_CLASSES, pretrained=False)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    n = sum(p.numel() for p in model.parameters())
    size_mb = n * 4 / (1024 * 1024)
    print(f'Model size: {size_mb:.2f} MB ({n:,} params)')

    dummy = torch.randn(1, 3, 224, 224)
    Path('models').mkdir(exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        opset_version=14,
    )
    print(f'Exported {out_path}')


if __name__ == '__main__':
    main()
