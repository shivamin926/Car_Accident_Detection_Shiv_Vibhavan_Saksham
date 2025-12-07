# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %%
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
from torchvision import models
from tqdm import tqdm, trange
import time

# %%
import torch, torchvision
print("torch.version:", torch.version)
print("torch.version.cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("current device index:", torch.cuda.current_device())
    try:
        print("device name:", torch.cuda.get_device_name(0))
    except Exception as e:
        print("get_device_name failed:", e)

# %%
# Define transforms
transform = transforms.Compose([
    transforms.Resize((300, 250)),       # Resize images
    transforms.ToTensor(),               # Convert to tensor (0–1 float)
])

# Load datasets
train_dataset = datasets.ImageFolder(
    root="Data/data1a/training",
    transform=transform
)

val_dataset = datasets.ImageFolder(
    root="Data/data1a/validation",
    transform=transform
)

# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2
)


# debugging 
# print("Classes:", train_dataset.classes)
# print("Train batches:", len(train_loader))
# print("Val batches:", len(val_loader))


# %%
class ResNetBinary(nn.Module):
    def __init__(self):
        super().__init__()

        # Load ResNet18 pretrained on ImageNet
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Get the number of features before the final layer
        num_features = self.model.fc.in_features

        # Replace the final fully connected layer for binary classification (2 outputs)
        self.model.fc = nn.Linear(num_features, 2)

    def forward(self, x):
        return self.model(x)



# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def train(model, dataset, epochs):
    optimizer = torch.optim.Adam(model.parameters())
    loss = nn.CrossEntropyLoss()
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    model = model.to(device)
    for epoch in trange(epochs):
        start = time.time()
        for (xs, targets) in tqdm(dataloader):
            xs, targets = xs.to(device), targets.to(device)
            ys = model(xs)
            optimizer.zero_grad()
            l = loss(ys, targets)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                acc = (ys.argmax(axis=1) == targets).sum() / xs.shape[0]
        duration = time.time() - start
        print("[%d] acc = %.2f loss = %.4f in %.2f seconds." % (epoch, acc.item(), l.item(), duration))


# %%
model = ResNetBinary()

# training on 4 pictures for testing
train(model, train_dataset, epochs=10)


# %%
def validate(model, dataloader):
    """Run model on validation DataLoader and return loss/accuracy."""
    model = model.to(device)
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total = 0
    correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for xs, targets in tqdm(dataloader):
            xs = xs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            ys = model(xs)
            loss = loss_fn(ys, targets)
            total_loss += loss.item() * xs.size(0)
            preds = ys.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += xs.size(0)
    avg_loss = total_loss / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    print(f"Validation - loss: {avg_loss:.4f}, accuracy: {acc:.4f} ({correct}/{total})")
    return {'loss': avg_loss, 'accuracy': acc, 'correct': correct, 'total': total}


# %%
try:
    _ = validate(model, val_loader)
except NameError:
    print('Model or val_loader not defined. Call validate(model, val_loader) after training or loading a model.')
