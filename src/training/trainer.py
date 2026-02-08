"""
Training utilities (same two-phase approach as before)
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm


class Trainer:
    def __init__(self, model, device, save_dir='models', save_best_as='best_model.pth'):
        self.model = model.to(device)
        self.device = device
        self.save_dir = save_dir
        self.save_best_as = save_best_as
        os.makedirs(save_dir, exist_ok=True)
        self.best_val_acc = 0.0

    def train_epoch(self, train_loader, criterion, optimizer):
        self.model.train()
        running_loss = 0.0
        correct, total = 0, 0
        for images, labels in tqdm(train_loader, desc='Training'):
            images, labels = images.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            out = self.model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, pred = torch.max(out, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
        return running_loss / len(train_loader), 100 * correct / total

    def validate(self, val_loader, criterion):
        self.model.eval()
        running_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Validating'):
                images, labels = images.to(self.device), labels.to(self.device)
                out = self.model(images)
                loss = criterion(out, labels)
                running_loss += loss.item()
                _, pred = torch.max(out, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
        return running_loss / len(val_loader), 100 * correct / total

    def train(self, train_loader, val_loader, epochs, lr=0.001, phase='classifier', patience=10):
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        # For ResNet: classifier phase trains only backbone.fc; for SmallDefectNet train all params
        if phase == 'classifier' and hasattr(self.model, 'backbone'):
            params = self.model.backbone.fc.parameters()
        else:
            params = self.model.parameters()
        optimizer = optim.Adam(params, lr=lr, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        best_val_acc = 0.0
        patience_counter = 0
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            val_loss, val_acc = self.validate(val_loader, criterion)
            scheduler.step(val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                }, os.path.join(self.save_dir, self.save_best_as))
                patience_counter = 0
            else:
                patience_counter += 1
            print(f'Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | Best: {best_val_acc:.2f}%')
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
        return best_val_acc
