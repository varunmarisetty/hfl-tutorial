import flwr as fl
import torch
import sys
import config
from utils import Net, train, test, load_client_data 

torch.manual_seed(seed=config.SEED)

class Client(fl.client.NumPyClient):
    def __init__(self, cid):
        self.cid = cid
        self.model = Net().to(config.DEVICE)
        
        # Load data specific to this client ID
        self.trainloader, self.testloader = load_client_data(int(cid))

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config_dict):
        # Load the received parameters into the model
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

        # Train the model
        train(self.model, self.trainloader, epochs=config.EPOCHS_PER_ROUND)

        # Return updated parameters, number of samples, and metrics
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

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