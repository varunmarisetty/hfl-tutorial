import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner
import config

# --- 1. MODEL DEFINITION ---
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(32 * 5 * 5, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32,1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        # Flatten output for binary classification and adjust for multiclass if needed
        return x.view(-1)

# --- 2. Train, Test LOOPS ---
def train(model, train_loader, epochs=1):
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE, momentum=config.MOMENTUM)
    model.train()
    for _ in range(epochs):
        for batch in train_loader:
            images = batch["image"].to(config.DEVICE)
            labels = batch["label"].float().to(config.DEVICE)
            
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
            images = batch["image"].to(config.DEVICE)
            labels = batch["label"].float().to(config.DEVICE)
            
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
    return loss / len(test_loader.dataset), correct / len(test_loader.dataset)

# --- 3. DATA UTILS ---
def apply_transforms(batch):
    transforms = Compose([ToTensor(), Normalize(mean=[.5], std=[.5])])
    batch["image"] = [transforms(img) for img in batch["image"]]
    return batch

def get_partitioner():
    """Factory function to create the requested partitioner."""
    if config.PARTITION_TYPE == "iid":
        return IidPartitioner(num_partitions=config.NUM_CLIENTS)
    else:
        return DirichletPartitioner(
            num_partitions=config.NUM_CLIENTS,
            partition_by="label",
            alpha=config.ALPHA,
            min_partition_size=config.MIN_PARTITION_SIZE,
            self_balancing=True
        )

# Initialize dataset once
import datasets
datasets.disable_caching()

fds = FederatedDataset(
    dataset=config.DATASET_NAME,
    subset=config.SUBSET,
    partitioners={"train": get_partitioner()},
    trust_remote_code=True,
)

def load_client_data(cid: int):
    """Loads the specific partition for a client ID."""
    # Load partition for this client
    partition = fds.load_partition(cid, "train")
    partition = partition.with_transform(apply_transforms)
    
    # Load global test set
    test_split = fds.load_split("test").with_transform(apply_transforms)
    
    trainloader = DataLoader(partition, batch_size=config.BATCH_SIZE, shuffle=True)
    testloader = DataLoader(test_split, batch_size=config.BATCH_SIZE)
    
    return trainloader, testloader