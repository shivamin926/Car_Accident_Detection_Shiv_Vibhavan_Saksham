import torch
import torch.nn as nn
from torchvision import models, transforms

# ---------------------------------------------------
# IMAGE TRANSFORMS (Used in both training & inference)
# ---------------------------------------------------
# Make sure this matches EXACTLY what you used when training
transform = transforms.Compose([
    transforms.Resize((300, 250)),   # Resize images
    transforms.ToTensor(),           # Convert to tensor
])


# ---------------------------------------------------
# MODEL DEFINITION
# ---------------------------------------------------
class ResNetBinary(nn.Module):
    def __init__(self):
        super().__init__()

        # Load pretrained ResNet18
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Replace final FC layer for binary classification
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 2)

    def forward(self, x):
        return self.model(x)


# ---------------------------------------------------
# TRAINING UTILITIES (Optional)
# ---------------------------------------------------
# These are here so you can train in a separate script,
# but they DO NOT RUN when this file is imported.
# ---------------------------------------------------

def train_model(model, train_loader, epochs, device):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for xs, targets in train_loader:
            xs, targets = xs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(xs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == targets).sum().item()
            total += targets.size(0)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}, Accuracy: {correct/total:.4f}")


def validate_model(model, val_loader, device):
    loss_fn = nn.CrossEntropyLoss()
    model = model.to(device)
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for xs, targets in val_loader:
            xs, targets = xs.to(device), targets.to(device)
            outputs = model(xs)
            loss = loss_fn(outputs, targets)

            total_loss += loss.item()
            correct += (outputs.argmax(1) == targets).sum().item()
            total += targets.size(0)

    print(f"Validation - Loss: {total_loss/total:.4f}, Accuracy: {correct/total:.4f}")
    return correct / total


# ---------------------------------------------------
# OPTIONAL: Only run TRAINING if this file is executed directly
# ---------------------------------------------------
if __name__ == "__main__":
    print("Run training from a separate script. This prevents accidental retraining during import.")