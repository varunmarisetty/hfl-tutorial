# Federated Learning on MedMNIST with Flower 🌼

This project implements a basic Federated Learning (FL) system using [Flower](https://flower.ai/) and [PyTorch](https://pytorch.org/). It simulates multiple clients training a CNN on the [MedMNIST v2](https://medmnist.com/) dataset (specifically `pneumoniamnist`) in a parallel, distributed manner on a single machine.

## 📂 Project Structure

* **`config.py`**: Central configuration for hyperparameters (learning rate, batch size), system settings (device, server address), and data partitioning.
* **`utils.py`**: Shared utilities including:
    * **`Net`**: The CNN model architecture.
    * **Data Loading**: Logic to download and partition MedMNIST using `flwr-datasets`.
    * **Training/Testing Loops**: Standard PyTorch train and evaluate functions.
* **`server.py`**: The FL Server. It manages the training rounds, aggregates updates from clients, and performs **centralized evaluation** on the global model using a held-out test set.
* **`client.py`**: The FL Client. Connects to the server, trains on its specific local data partition, and sends model updates back.
* **`run_exp.py`**: An orchestration script that launches the server and multiple client processes in parallel.
    * **Windows**: Opens separate tabs in Windows Terminal.
    * **Linux/Mac**: Runs processes in the background.
* **`dashboard.py`**: A real-time [Streamlit](https://streamlit.io/) dashboard to visualize global accuracy and loss.

## 🛠️ Prerequisites

Ensure you have Python 3.11+ installed. Install the required dependencies:

```bash
pip install flwr==1.18.0 torch torchvision flwr-datasets[vision]==0.5.0 medmnist streamlit pandas
```

## ⚙️ Configuration

Modify config.py to change experiment settings:
```python

# Key parameters in config.py
NUM_CLIENTS = 2           # Number of clients to simulate
NUM_ROUNDS = 10           # Total FL rounds
PARTITION_TYPE = "dirichlet" # 'iid' (random) or 'dirichlet' (non-IID/heterogeneous)
ALPHA = 0.5               # Heterogeneity level (lower = more heterogeneous)
DEVICE = "cpu"            # 'cuda' or 'cpu'
```

## 🚀 How to Run

### 1. Start the Experiment
Run the orchestration script to launch the server and all clients automatically:

```bash
python run_exp.py
```

### 2. View the Dashboard
Open a new terminal and start the Streamlit dashboard to monitor training progress in real-time:

```bash
streamlit run dashboard.py
```

This will open a web page (usually at http://localhost:8501) showing charts for Global Accuracy and Loss.

### 📊 Features

Realistic Simulation: Unlike simple loops, this project uses subprocess to spawn distinct processes for every client, mimicking a real network deployment.

Heterogeneous Data: Supports non-IID data partitioning (via Dirichlet distribution), allowing you to simulate real-world scenarios where data is not evenly distributed across hospitals.

Centralized Evaluation: The server keeps a global test set to evaluate the "true" performance of the aggregated model after every round.

Live Metrics: Metrics are logged to fl_metrics.csv and visualized instantly.

### 🐛 Troubleshooting

Shape Mismatch: If you encounter errors regarding target size vs input size, ensure utils.py includes .view(-1) in the model's forward pass to flatten the output.

Connection Refused: If clients fail to connect, check SERVER_ADDRESS in config.py. Try changing 0.0.0.0:8080 to 127.0.0.1:8080.