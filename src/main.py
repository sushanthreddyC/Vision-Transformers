
## Standard libraries
import os
import json
import math
import random
import numpy as np 
import copy
import time
import pandas as pd
## Imports for plotting
# %matplotlib inline
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
sns.set_theme()

## Progress bar
from tqdm.notebook import tqdm

## typing
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

## PyTorch Torchvision
import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms


# PyTorch Lightning
import pytorch_lightning as pl

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

# Import tensorboard
# %load_ext tensorboard


from utils import *
from data import *
from data_transformations import *
from vit_trainer import ViT
from IdentityFormer_trainer import ViT_Identity
from ConvFormer_trainer import ViT_ConvFormer

def  download_CIFAR10_data(DATASET_PATH, train_transform,test_transform):
    # Loading the training dataset. We need to split it into a training and validation part
    # We need to do a little trick because the validation set should not use the augmentation.
    train_dataset = CIFAR10(root=DATASET_PATH, train=True, transform=train_transform, download=True)
    val_dataset = CIFAR10(root=DATASET_PATH, train=True, transform=test_transform, download=True)

    # Loading the test set
    test_set = CIFAR10(root=DATASET_PATH, train=False, transform=test_transform, download=True)
    return train_dataset, val_dataset,test_set

def train_model(max_epochs,model_name,model_type, train_loader, val_loader,test_loader, **kwargs):
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, model_name), 
                         accelerator="auto", devices="auto",
                         max_epochs=max_epochs,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
                                    LearningRateMonitor("epoch")],
                         check_val_every_n_epoch=10)
    trainer.logger._log_graph = True         # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, model_name+".ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        model = model_type.load_from_checkpoint(pretrained_filename) # Automatically loads the model with the saved hyperparameters
    else:
        pl.seed_everything(42) # To be reproducable
        model = model_type(train_loader,**kwargs)
        trainer.fit(model, train_loader, val_loader)
        model = model_type.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # Load best checkpoint after training

    # val_accs = trainer.logged_metrics['val_acc'] if 'val_acc' in trainer.logged_metrics else []
    
    # Test best model on validation and test set
    val_result = trainer.test(model, val_loader, verbose=False)
    test_result = trainer.test(model, test_loader, verbose=False)
    result = {"test": test_result[0]['test_acc_epoch'], "val": val_result[0]["test_acc_epoch"]}

    return model, test_result


def train_vit(train_loader, val_loader, test_loader):
    val_acc={}
    model, results = train_model(model_type=ViT, model_name="VIT", train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,model_kwargs={
                                    'embed_dim': 128,
                                    'hidden_dim': 256,
                                    'num_heads': 8,
                                    'num_layers': 6,
                                    'patch_size': 8,
                                    'num_channels': 3,
                                    'num_patches': 16,
                                    'num_classes': 10,
                                    'dropout': 0.0
                                },
                                lr=3e-4,
                                max_epochs=100)
    print("ViT results", results)

def train_conv_former(train_loader, val_loader, test_loader):
    # Experiment with Identity Former
    model, results = train_model(model_type=ViT_ConvFormer, model_name="ConvFormer", train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,model_kwargs={
                                    'embed_dim': 128,
                                    'hidden_dim': 256,
                                    'num_heads': 8,
                                    'num_layers': 4,
                                    'patch_size': 8,
                                    'num_channels': 3,
                                    'num_patches': 16,
                                    'num_classes': 10,
                                    'dropout': 0.0
                                },
                                lr=3e-4,
                                max_epochs=10)
    print("ViT results", results)

def train_identity_former(train_loader, val_loader, test_loader):
# Experiment with Identity Former
    model, results = train_model(model_type=ViT_Identity, model_name="IdentityFormer",train_loader=train_loader, val_loader=val_loader, test_loader=test_loader ,model_kwargs={
                                    'embed_dim': 128,
                                    'hidden_dim': 256,
                                    'num_heads': 8,
                                    'num_layers': 4,
                                    'patch_size': 8,
                                    'num_channels': 3,
                                    'num_patches': 16,
                                    'num_classes': 10,
                                    'dropout': 0.0
                                },
                                lr=3e-4,
                                max_epochs=10)
    print("ViT results", results)



def _main_():
    random_seed(seed=seed, deterministic=True)
    # Loading the training dataset. We need to split it into a training and validation part
    # We need to do a little trick because the validation set should not use the augmentation.
    train_dataset,val_dataset,test_set = download_CIFAR10_data(DATASET_PATH,train_transform,test_transform)
    train_set, _ = torch.utils.data.random_split(train_dataset, [45000, 5000], generator=torch.Generator().manual_seed(42))
    _, val_set = torch.utils.data.random_split(val_dataset, [45000, 5000], generator=torch.Generator().manual_seed(42))


    # Create data loaders for later. Adjust batch size if you have a smaller GPU
    train_loader = data.DataLoader(train_set, batch_size=128, shuffle=True, drop_last=True, pin_memory=True, num_workers=3)
    val_loader = data.DataLoader(val_set, batch_size=128, shuffle=False, drop_last=False, num_workers=3)
    test_loader = data.DataLoader(test_set, batch_size=128, shuffle=False, drop_last=False, num_workers=3)
