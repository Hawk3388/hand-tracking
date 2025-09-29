import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os
import json
from collections import defaultdict
import random

class FrameMLP(nn.Module):
    def __init__(self, input_dim=63, hidden_dims=[128, 128, 64], num_classes=36):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5)
            ]
            prev = h
        self.net = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev, num_classes)

    def forward(self, x):
        # x: [batch, 63]
        feat = self.net(x)
        return self.classifier(feat)  # logits

class FrameMLP2(nn.Module):
    """Optimierte Architektur: mehr Schichten, LeakyReLU, Dropout-Variation, LayerNorm"""
    def __init__(self, input_dim=63, hidden_dims=[512, 256, 128, 128, 64], num_classes=36):
        super().__init__()
        layers = []
        prev = input_dim
        for i, h in enumerate(hidden_dims):
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
            # Weniger Dropout in frühen Schichten, mehr in späten
            # dropout_rate = 0.2
            # layers.append(nn.Dropout(dropout_rate))
            prev = h
        self.net = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev, num_classes)
    def forward(self, x):
        feat = self.net(x)
        return self.classifier(feat)

class ASLFolderDataset(Dataset):
    def __init__(self, root_dir):
        # Nur Ordner mit einem Buchstaben als Label nehmen
        class_dirs = [d for d in sorted(os.listdir(root_dir)) if os.path.isdir(os.path.join(root_dir, d)) and len(d) == 1 and d.isalpha()]
        self.class_to_idx = {cls: i for i, cls in enumerate(class_dirs)}
        self.samples = []
        for cls in self.class_to_idx:
            class_dir = os.path.join(root_dir, cls)
            for fname in os.listdir(class_dir):
                if fname.endswith('.json'):
                    self.samples.append((os.path.join(class_dir, fname), self.class_to_idx[cls]))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with open(path, 'r') as f:
            landmarks = json.load(f)
        # landmarks: list of 21 dicts mit x,y,z
        flat = []
        for lm in landmarks:
            flat.extend([lm['x'], lm['y'], lm['z']])
        return torch.tensor(flat, dtype=torch.float32), label

def collate_fn(batch):
    x = torch.stack([item[0] for item in batch])
    y = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return x, y

if __name__ == "__main__":
    # Hyperparameter
    batch_size = 32  # kleiner für stabileres Training
    epochs = 200
    lr = 1e-3
    patience = 30  # Early Stopping

    # Modell und Device
    # model = FrameMLP()  # Original
    model = FrameMLP2()   # Optimiertes Modell
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=7, verbose=True)
    criterion = torch.nn.CrossEntropyLoss()

    # Eigenes Dataset laden und festen Validation-Split erzeugen
    ds_full = ASLFolderDataset("./asl/dataset")
    # Mapping: label -> list of indices
    label_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(ds_full.samples):
        label_to_indices[label].append(idx)
    val_indices = []
    train_indices = []
    random.seed(42)  # Für Reproduzierbarkeit
    # Für jeden Buchstaben mindestens ein Sample ins Validation-Set
    for indices in label_to_indices.values():
        random.shuffle(indices)
        val_indices.append(indices[0])
        train_indices.extend(indices[1:])
    # Restliche Validation-Samples auffüllen (bis 10% der Daten)
    needed = int(0.1 * len(ds_full)) - len(val_indices)
    if needed > 0:
        remaining = list(set(range(len(ds_full))) - set(val_indices))
        random.shuffle(remaining)
        val_indices.extend(remaining[:needed])
        train_indices = list(set(range(len(ds_full))) - set(val_indices))
    ds_train = torch.utils.data.Subset(ds_full, train_indices)
    ds_val = torch.utils.data.Subset(ds_full, val_indices)
    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    loader_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Trainingsloop
    best_val_acc = 0
    best_epoch = 0
    val_accs = []
    epochs_no_improve = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for x, y in loader_train:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        train_loss = total_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in loader_val:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * x.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)
        val_loss = val_loss / val_total if val_total > 0 else 0
        val_acc = val_correct / val_total if val_total > 0 else 0
        val_accs.append(val_acc)
        scheduler.step(val_acc)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

        # Early Stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            epochs_no_improve = 0
            torch.save(model.state_dict(), "./asl/frame_mlp_asl.pt")
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping nach {epoch+1} Epochen.")
            break

    print("Modell wurde als 'frame_mlp_asl.pt' gespeichert.")
    print(f"Beste Validierungsgenauigkeit: {best_val_acc:.4f} in Epoche {best_epoch}")
