# Training Parameters
NUM_CLIENTS = 2
NUM_ROUNDS = 10
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
METRICS_FILE = "fl_metrics.csv"
DEVICE = "cpu"
SERVER_ADDRESS = "0.0.0.0:8080"