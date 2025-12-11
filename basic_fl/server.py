import flwr as fl
import csv
import os
from fl_utils import Net, test, load_partition # Used for server-side evaluation if needed

# Setup Logging File
CSV_FILE = "fl_metrics.csv"
if os.path.exists(CSV_FILE):
    os.remove(CSV_FILE)
with open(CSV_FILE, "w") as f:
    f.write("round,accuracy,loss\n")

def write_metrics(rnd, accuracy, loss):
    with open(CSV_FILE, "a") as f:
        f.write(f"{rnd},{accuracy},{loss}\n")

# Custom Strategy to Aggregate Metrics
class LoggedStrategy(fl.server.strategy.FedAvg):
    def aggregate_evaluate(self, server_round, results, failures):
        # Call original aggregation
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)
        
        if aggregated_metrics:
            acc = aggregated_metrics['accuracy']
            print(f"Round {server_round} - Aggregated Accuracy: {acc}")
            write_metrics(server_round, acc, aggregated_loss)
            
        return aggregated_loss, aggregated_metrics

# Aggregation function for weighted average
def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

# Start Server
strategy = LoggedStrategy(
    evaluate_metrics_aggregation_fn=weighted_average,
    min_fit_clients=2,  # Wait for 2 clients
    min_evaluate_clients=2,
    min_available_clients=2,
)

print("Starting Server... (Open dashboard.py in another terminal)")
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy
)