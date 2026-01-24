import flwr as fl
import torch
import sys
import config
import os
from utils import Net, train, test, load_client_data 

torch.manual_seed(seed=config.SEED)

class Client(fl.client.NumPyClient):
    def __init__(self, cid):
        self.cid = cid
        self.model = Net().to(config.DEVICE)
        
        # Load data specific to this client ID
        self.trainloader, self.testloader = load_client_data(int(cid))

        # Initialize client-specific log file
        self.log_path = config.get_client_log_path(self.cid)
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w") as f:
                f.write("round,accuracy,loss\n")
        
        self.current_round = 0

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config_dict):
        # Increment internal round counter
        self.current_round += 1

        # Load the received parameters into the model
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

        # Train the model
        train_loss, train_acc = train(self.model, self.trainloader, epochs=config.EPOCHS_PER_ROUND)

        # Save the client's model
        model_path = config.get_client_model_path(self.cid)
        torch.save(self.model.state_dict(), model_path)
        
        # Log training metrics locally
        with open(self.log_path, "a") as f:
            f.write(f"{self.current_round},{train_acc},{train_loss}\n")
        
        print(f"Client {self.cid} training (Round {self.current_round}): Accuracy: {train_acc:.4f}, Loss: {train_loss:.4f}")

        # Return updated parameters, number of samples, and metrics
        return self.get_parameters(config={}), len(self.trainloader.dataset), {"accuracy": train_acc}

    def evaluate(self, parameters, config_dict):
        # Load the received parameters into the model
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
        
        # Evaluate the model
        loss, accuracy = test(self.model, self.testloader)
        
        # Return loss, number of samples, and metrics
        return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}

if __name__ == "__main__":
    # Check if Client ID is provided
    if len(sys.argv) < 2:
        print("Usage: python client.py <client_id>")
        sys.exit(1)
        
    client_id = sys.argv[1]
    
    # Start the client
    # Note: If 0.0.0.0 doesn't work for connecting on your OS, try changing config to 127.0.0.1
    print(f"Starting Client {client_id}...")
    fl.client.start_numpy_client(server_address=config.SERVER_ADDRESS, client=Client(client_id).to_client())