import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import Compose, ToTensor, Normalize

DEVICE = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. MODEL DEFINITION (Unchanged) ---
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(32 * 5 * 5, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# --- 2. TRAINING FUNCTIONS (Unchanged) ---
def train(model, train_loader, epochs=1):
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    model.train()
    for batch in train_loader:
        # Note: formatting matches HF dataset output structure
        images = batch["image"].to(DEVICE)
        labels = batch["label"].float().to(DEVICE)
        
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()

def test(model, test_loader):
    criterion = nn.BCELoss()
    model.eval()
    loss, correct = 0.0, 0
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(DEVICE)
            labels = batch["label"].float().to(DEVICE)
            
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
    return loss / len(test_loader.dataset), correct / len(test_loader.dataset)

# --- 3. TRANSFORMS (New) ---
# This converts the raw HF Data (PIL Images) to Tensors
def apply_transforms(batch):
    transforms = Compose([ToTensor(), Normalize(mean=[.5], std=[.5])])
    
    # Apply to image column
    batch["image"] = [transforms(img) for img in batch["image"]]
    
    # MedMNIST labels come as lists like [1], we need simple integers/floats
    # We flatten the label list if necessary
    # Note: MedMNIST v2 on HF typically returns labels as integers inside a list
    return batch