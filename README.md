# Vision Transformers (ViT) Project

## Overview

This project involves the implementation and training of Vision Transformers (ViT) for image classification tasks. Vision Transformers have been successfully applied to a range of computer vision problems, demonstrating impressive performance.

## Project Structure

- `Project_02_ViTs_schilla_results.ipynb`: Jupyter notebook containing the experiment results, model architecture, and training procedures.
- `run_jupyter.sh`: Shell script to initiate the Jupyter notebook environment.

## Getting Started

### Prerequisites

- Python 3.x
- PyTorch
- Transformers library
- Jupyter Notebook

### Installation

To set up the project environment:

1. Clone the repository: git clone https://github.com/sushanthreddyC/Vision-Transformers.git
2. Install the required packages: source './ece/setup.sh'


### Running the Project

To start the Jupyter Notebook server:


- source ./run_jupyter.sh

## Results
- The training process has been logged using TensorBoard. The following key metrics are tracked:

- Training Accuracy
- Validation Accuracy
- Training Loss
- Validation Loss
- From the experiments, lightning_logs/version_4 shows the highest validation accuracy, suggesting model improvements over iterations.
