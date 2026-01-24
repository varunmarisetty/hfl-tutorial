import flwr as fl
import os
import config 
from utils import Net, test, load_client_data
import torch

torch.manual_seed(seed=config.SEED)

import shutil

# Ensure directories exist and are clean for a new experiment
if os.path.exists(config.MODELS_DIR):
    shutil.rmtree(config.MODELS_DIR)
if os.path.exists(config.LOGS_DIR):
    shutil.rmtree(config.LOGS_DIR)

os.makedirs(config.MODELS_DIR, exist_ok=True)
os.makedirs(config.LOGS_DIR, exist_ok=True)

# Initialize global log
with open(config.GLOBAL_LOG_PATH, "w") as f:
    f.write("round,accuracy,loss\n")

def write_metrics(rnd, accuracy, loss):
    with open(config.GLOBAL_LOG_PATH, "a") as f:
        f.write(f"{rnd},{accuracy},{loss}\n")

def evaluate_fn(server_round: int, parameters: fl.common.NDArrays, config_dict: dict):
    """This function is called by the server after every round."""
    _, test_loader = load_client_data(0)
    model = Net()
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)
    
    loss, accuracy = test(model, test_loader)
    print(f"Server-side evaluation (Round {server_round}): Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")
    write_metrics(server_round, accuracy, loss)
    
    torch.save(model.state_dict(), config.GLOBAL_MODEL_PATH)
    print(f"Model saved to {config.GLOBAL_MODEL_PATH}")
    
    return loss, {"accuracy": accuracy}

if __name__ == "__main__":
    print("Starting Server...")
    strategy = fl.server.strategy.FedAvg(
        evaluate_fn=evaluate_fn,
        min_fit_clients=config.NUM_CLIENTS,
        min_evaluate_clients=config.NUM_CLIENTS,
        min_available_clients=config.NUM_CLIENTS,
    )
    fl.server.start_server(
        server_address=config.SERVER_ADDRESS, 
        config=fl.server.ServerConfig(num_rounds=config.NUM_ROUNDS),
        strategy=strategy
    )

