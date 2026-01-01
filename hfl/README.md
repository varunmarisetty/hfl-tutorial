# 📚 HFL Tutorial: Hierarchical Federated Learning

This project implements a **Hierarchical Federated Learning (HFL)** system using the [Flower (flwr)](https://flower.dev/) framework.

Unlike traditional Federated Learning, where all clients connect directly to a single central server, this architecture introduces an intermediate layer of **Edge Servers** to improve scalability.

---

## 🏗️ Architecture Topology

In this topology, the **Edge Server** acts as a **model relay** with a dual role:

1. **Server Role**
   - Waits for connections from leaf **Clients**
   - Aggregates their model updates
   - Produces a temporary **edge-level global model**

2. **Client Role**
   - Acts as a client to the **Central Server**
   - Sends the aggregated edge model upward as a single update

This hierarchical structure mirrors real-world deployments such as IoT, and edge-cloud systems.

---

## 🚀 Quick Start

### 1️⃣ Prerequisites

- Python **3.11+**
- Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Configuration  
- **topo.yml:** Defines the network hierarchy (which clients connect to which edges).  
- **config.py:** Sets hyperparameters such as number of rounds, learning rate, and minimum clients per server.

### 3. Run Simulation  
The `simulate.py` script orchestrates the entire network based on your topology configuration.

- **Windows:** Opens separate terminal tabs for each node.  
- **Linux:** Runs processes in the background.

### 4. Dashboard  
Monitor training progress (Loss/Accuracy) for the **Central Server**, **Edge Servers**, and individual **Clients** in real time.

---

## Orchestration

### 1. Experiment Simulator - `simulate.py`
This script bootstraps the entire HFL setup.

**Key methods:**
- `spawn_processes()`  
  - **Reads** `topo.yml`.  
  - **Dynamic Port Assignment:** Uses `get_free_port()` if ports are unspecified to avoid collisions.  
  - **Client Calculation:** Determines how many clients each edge server should wait for and passes it as an argument to `edge_server.py`.  
  - **Process Launching:** Uses `subprocess.Popen` (Linux) or `start cmd` (Windows) to launch the Central Server, Edge Servers, then Clients sequentially.

---

### 2. Central or Cloud Node - `central_server.py`
Represents the root of the hierarchy (Cloud).

**Classes & Methods:**
- `FedAvg` (inherits from `flwr.server.strategy.FedAvg`):
  - `aggregate_evaluate`: Aggregates Edge Servers' accuracy/loss metrics.
  - `evaluate`: Evaluates the global model on a hold-out test set after each round.
- `main`: Initializes the server with `min_fit_clients` based on the number of Edge Servers.

---

### 3. Edge Server - `edge_server.py`
Handles the **dual Server/Client role**.

**Core components:**
- `EdgeStrategy`:  
  - Strategy for the local edge server.  
  - `aggregate_fit`: Collects updates from leaf clients and stores aggregated parameters in `shared_state["aggregated_model"]`.

- `EdgeClient`:  
  - Acts as a client to the Central Server.  
  - `fit(parameters, config)`:  
    - Receives the Global Model.  
    - Spawns a subprocess (`run_edge_server`) to start the local Flower server.  
    - Waits until local training completes, retrieves aggregated results from `shared_state`, and sends them to the Central Server.
    - 
---

### 4. End Clients - `client.py`
Represents leaf clients performing local training.

**Class:** `FlowerClient`
- `fit`:  
  - Validates the received model.  
  - Runs local training (`utils.train` or `utils.train_fedprox`).  
  - Returns updated weights to the parent Edge Server.
- `evaluate`: Tests the model on the local validation set.
- `create_client`: Loads data partitions for this client and exports JSON summaries for debugging.

---

### 5. Utilities - `utils.py`
Shared helper functions supporting training and data operations.

**Main functions:**
- `load_datasets`:  
  - Uses `flwr_datasets` for dataset download and partitioning (e.g., MNIST, CIFAR10, etc).  
  - Supports IID (random shuffle) and Non-IID (Dirichlet/Pathological) splits.
- `train`: Standard PyTorch training loop.
- `train_fedprox`: Modified training loop with a proximal term (μ) to stabilize convergence under heterogeneous data.

---

### 6. Visualization - `dashboard.py`
A **Streamlit** app providing real-time insights.

**Functions:**
- `render_central_server_logs`: Plots global Loss and Accuracy.
- `render_edge_server_logs`: Dynamically discovers and plots metrics for active Edge Servers.
- `render_clients_logs`: Visualizes training/testing performance of individual clients across datasets.

---

## 🧩 Summary
The HFL system demonstrates a scalable, modular architecture for multi-tier federated learning.  
It mirrors real-world edge-cloud hierarchies while maintaining flexible orchestration and monitoring across all layers.

