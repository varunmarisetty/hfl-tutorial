NUM_ROUNDS = 10
TOPOLOGY_FILE = "topo.yml"

NUM_CLIENTS = 4
MIN_CLIENTS_PER_EDGE = 2

MODEL = "medmnist_cnn"
DATASET = "albertvillanova/medmnist-v2"
SUBSET = "pneumoniamnist"

SEED = 42

LOCAL_EPOCHS = 1
BATCH_SIZE = 16
PARTITIONER = "iid"
DIRICHLET_ALPHA = 0.1
NUM_CLASSES_PER_PARTITION = 3  # used in pathological partitioning (limit label)
NUM_CLASSES = 10  # total number of classes in the dataset
LOSS_FUNCTION = "binary_cross_entropy"  # options: "cross_entropy", "binary_cross_entropy"

TRAINING_LEARNING_RATE = 5 * 1e-4
TRAINING_WEIGHT_DECAY = 1e-4

TRAINING_SCHEDULER_STEP_SIZE = 10
TRAINING_SCHEDULER_GAMMA = 0.1
TRAINING_STRATEGY = "fedavg"  # options: "fedavg", "fedprox"
FedProx_MU = 0.01

SPLIT=PARTITIONER

if PARTITIONER == "dirichlet":
    SPLIT=f"{SPLIT}_{DIRICHLET_ALPHA}"
elif PARTITIONER == "pathological":
    SPLIT=f"{SPLIT}_{NUM_CLASSES_PER_PARTITION}"


EXPERIMENT_NAME = f"{DATASET.split('/')[-1]}-{NUM_CLIENTS}c-{MODEL}-{SPLIT}"
