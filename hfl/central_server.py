import importlib
import json
import flwr as fl
from flwr.server import ServerConfig
import argparse
import matplotlib.pyplot as plt
import numpy as np
from config import NUM_ROUNDS, MODEL, SEED
from logger import Logger

from utils import set_parameters, test, load_datasets
from flwr.common import parameters_to_ndarrays, FitIns

np.random.seed(SEED)

parser = argparse.ArgumentParser(description="Start the Flower central server.")
parser.add_argument(
    "address", help="Server address in the format host:port (e.g., 0.0.0.0:8081)"
)
parser.add_argument(
    "--exp_id",
    type=str,
    help="The experiment ID for the dashboard",
)
parser.add_argument(
    "--min_edges", 
    type=int, 
    default=2, 
    help="Min edges required"
    )

args = parser.parse_args()

logger = Logger(
    subfolder="central",
    file_path="central_server.log",
    headers=["round", "loss", "accuracy"],
)

server_address = args.address

class FedAvg(fl.server.strategy.FedAvg):
    def __init__(self):
        super().__init__(
            min_fit_clients=int(args.min_edges),
            min_available_clients=int(args.min_edges), # total edges
            on_fit_config_fn=lambda rnd: {"round": rnd}, # aggregate every round
            on_evaluate_config_fn=lambda rnd: {"round": rnd}, # evaluate every round
        )
        self.yi_per_group = {}  # store yi for each group/edge

    def aggregate_fit(self, rnd, results, failures):
        aggregated_parameters = super().aggregate_fit(rnd, results, failures)
        # print(aggregated_parameters, rnd, results, failures)

        return aggregated_parameters

    def configure_fit(self, server_round, parameters, client_manager, **kwargs):
        """
        Configure per-client fit instructions with yi.
        """
        # Get default instructions from FedAvg
        fit_instructions = super().configure_fit(
            server_round, parameters, client_manager, **kwargs
        )

        return fit_instructions

    def evaluate(self, server_round, parameters):
        if server_round == 0:
            print("Skipping evaluation for round 0")
            return super().evaluate(server_round, parameters)

        print(f"[Central Server] Evaluate round {server_round}")

        param_arrays = parameters_to_ndarrays(parameters)
        if all(np.allclose(p, 0) for p in param_arrays):
            print("[Warning] All parameters are zero! Skipping evaluation.")
            return super().evaluate(server_round, parameters)

        model_module = importlib.import_module(f"models.{MODEL}")
        net = model_module.Net()

        # print(parameters_to_ndarrays(parameters)[0][0][0][0])
        set_parameters(net, param_arrays)
        _, _, testloader = load_datasets()  # full dataset for evaluation
        loss, accuracy = test(net, testloader)
        logger.log(
            {
                "round": server_round,
                "loss": loss,
                "accuracy": accuracy,
            }
        )

        print(
            f"[Central Server] Evaluate Round {server_round}: Loss = {loss}, Accuracy = {accuracy}"
        )
        return super().evaluate(server_round, parameters)

    def aggregate_evaluate(self, server_round, results, failures):
        """Log loss values after each round."""

        if not results:
            return None, {}

        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )

        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]
        aggregated_accuracy = sum(accuracies) / sum(examples)
        # print(list(zip(accuracies, examples)))

        print(
            f"[Central Server] Round {server_round}: Average Loss = {aggregated_loss}"
        )
        print(
            f"[Central Server] Round {server_round}: Average Accuracy = {sum(accuracies) / sum(examples)}"
        )

        return float(aggregated_loss), {"accuracy": float(aggregated_accuracy)}
    

if __name__ == "__main__":
    config = ServerConfig(num_rounds=NUM_ROUNDS)
    strategy = FedAvg()
    print(f"Starting central server at {server_address}")
    fl.server.start_server(
        server_address=server_address, strategy=strategy, config=config
    )
