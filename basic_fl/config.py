import os

# Get the directory where config.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Training Parameters
NUM_CLIENTS = 2
NUM_ROUNDS = 20
BATCH_SIZE = 16
EPOCHS_PER_ROUND = 1
LEARNING_RATE = 0.01
MOMENTUM = 0.9

SEED = 42

# Data Parameters
DATASET_NAME = "albertvillanova/medmnist-v2"
SUBSET = "pneumoniamnist"
NUM_CLASSES = 2
PARTITION_TYPE = "dirichlet"  # 'iid' or 'dirichlet'
ALPHA = 0.5                   # Heterogeneity for dirichlet
MIN_PARTITION_SIZE = 10

# System Parameters
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

GLOBAL_MODEL_PATH = os.path.join(MODELS_DIR, "global_model.pth")
GLOBAL_LOG_PATH = os.path.join(LOGS_DIR, "global_metrics.csv")

def get_client_model_path(cid):
    return os.path.join(MODELS_DIR, f"client_{cid}.pth")

def get_client_log_path(cid):
    return os.path.join(LOGS_DIR, f"client_{cid}_metrics.csv")

# LEGACY (Keeping for compatibility until fully transitioned)
METRICS_FILE = GLOBAL_LOG_PATH
MODEL_PATH = GLOBAL_MODEL_PATH

DEVICE = "cpu"
SERVER_ADDRESS = "localhost:8080"
