import os
## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import Subset
import torch.optim as optim
import os
## PyTorch Torchvision
import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms
# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
DATASET_PATH = "./data"
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "./checkpoints/ece763_proj_02"
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

seed = 42


device = (
    torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
)
print("Using device", device)

# classes
classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
    )


