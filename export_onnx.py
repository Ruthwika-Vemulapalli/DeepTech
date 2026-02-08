"""
Export best PyTorch model to ONNX for deployment/portal
"""
import torch
from pathlib import Path
from src.models.resnet import ResNet50Classifier
from src.data.dataset import NUM_CLASSES

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ResNet50Classifier(num_classes=NUM_CLASSES, pretrained=False)
    ckpt = torch.load('models/best_model.pth', map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)
    Path('models').mkdir(exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        'models/defect_8class.onnx',
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        opset_version=14,
    )
    print('Exported models/defect_8class.onnx')


if __name__ == '__main__':
    main()
