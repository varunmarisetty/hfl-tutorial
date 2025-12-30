import flwr as fl
import os
import config 
from utils import Net, test, load_client_data
import torch

torch.manual_seed(seed=config.SEED)

# Use config for file path
if os.path.exists(config.METRICS_FILE):
    os.remove(config.METRICS_FILE)
with open(config.METRICS_FILE, "w") as f:
    f.write("round,accuracy,loss\n")

def write_metrics(rnd, accuracy, loss):
    with open(config.METRICS_FILE, "a") as f: # <--- UPDATED
        f.write(f"{rnd},{accuracy},{loss}\n")

def evaluate_fn(server_round: int, parameters: fl.common.NDArrays, config_dict: dict):
    """This function is called by the server after every round."""
    
    _, test_loader = load_client_data(0)
    # Instantiate the model
    model = Net()
    
    # Load the latest global weights into the model
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)
    
    # Run the test function on the server's test loader
    loss, accuracy = test(model, test_loader)
    
    # Log results
    print(f"Server-side evaluation (Round {server_round}): Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")
    write_metrics(server_round, accuracy, loss)
    
    return loss, {"accuracy": accuracy}


if __name__ == "__main__":
    print("Starting Server...")
    strategy = fl.server.strategy.FedAvg(
        evaluate_fn=evaluate_fn,  
        min_fit_clients=config.NUM_CLIENTS, # global model aggregation
        min_evaluate_clients=config.NUM_CLIENTS, # global model evaluation
        min_available_clients=config.NUM_CLIENTS, # total clients
    )
    fl.server.start_server(
        server_address=config.SERVER_ADDRESS, 
        config=fl.server.ServerConfig(num_rounds=config.NUM_ROUNDS),
        strategy=strategy
    )