import argparse
import flwr as fl
import torch
from torch.utils.data import DataLoader
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner
from flwr_datasets.visualization import plot_label_distributions
from fl_utils import Net, train, test, apply_transforms, DEVICE

# Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--server", type=str, default="127.0.0.1")
parser.add_argument("--id", type=int, required=True) # Unique ID (0, 1, 2...)
parser.add_argument("--partition", type=str, default="iid") # 'iid' or 'dirichlet'
parser.add_argument("--alpha", type=float, default=0.5) # Heterogeneity
args = parser.parse_args()

# --- 1. SETUP PARTITIONER ---
# This replaces all the complex manual slicing code
if args.partition == "iid":
    partitioner = IidPartitioner(num_partitions=6) # Assuming 6 devices
else:
    partitioner = DirichletPartitioner(
        num_partitions=6, 
        partition_by="label", 
        alpha=args.alpha,
        min_partition_size=10,
        self_balancing=True
    )

# Saves the label distribution of the partition created
fig, ax, df = plot_label_distributions(
    partitioner, 
    label_name="label", 
    partition_id=args.id, 
    plot_type="bar"
)
fig.savefig(f"client_{args.id}_distribution.png")

# --- 2. LOAD FEDERATED DATASET ---
# We pull 'medmnist-v2' directly from Hugging Face
fds = FederatedDataset(
    dataset="albertvillanova/medmnist-v2", 
    subset="pneumoniamnist", 
    partitioners={"train": partitioner}
)

print(f"Client {args.id}: Loading partition...")

# Load THIS client's slice
partition = fds.load_partition(args.id, "train")
partition = partition.with_transform(apply_transforms)

# Convert to PyTorch DataLoader
trainloader = DataLoader(partition, batch_size=32, shuffle=True)

# For testing, we can load the centralized test set (or a local slice)
test_split = fds.load_split("test").with_transform(apply_transforms)
testloader = DataLoader(test_split, batch_size=32)

# --- 3. FLOWER CLIENT ---
model = Net().to(DEVICE)

class MedClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

    def fit(self, parameters, config):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)

        train(model, trainloader, epochs=1)
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)
        
        loss, accuracy = test(model, testloader)
        return float(loss), len(testloader.dataset), {"accuracy": float(accuracy)}

print(f"Connecting to {args.server}...")
fl.client.start_numpy_client(server_address=f"{args.server}:8080", client=MedClient())