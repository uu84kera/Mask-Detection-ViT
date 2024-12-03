# Mask Detection using Vision Transformer (ViT)

## Project Structure

```
/Mask Detection using ViT
    /vit_pytorch
        __init__.py
        vit.py
        simple_vit.py
    train.py
    data_loader.py
    eval_metric.py
    /dataset
        /mfr2
        train.txt
        test.txt
        validate.txt
```

### Directory Details

- **vit_pytorch/**: Contains the implementation of Vision Transformer (ViT) models.
  - `__init__.py`: Used for module initialization.
  - `vit.py`: Implementation of the original Vision Transformer (ViT).
  - `simple_vit.py`: A simplified version of ViT used in this project for mask detection.

- **train.py**: The main script to train the Vision Transformer for mask detection. This script takes command-line arguments for training settings (e.g., learning rate, batch size) and outputs logs and checkpoints.

- **data_loader.py**: Implements the `SimpleVitDataset` class used for loading the dataset. It reads image paths and labels from the text files and applies necessary transformations for training.

- **eval_metric.py**: Provides utility functions to evaluate the trained model using various metrics like accuracy, specificity, sensitivity, and Cohen's Kappa.

- **dataset/**: Contains the dataset for training, validation, and testing.
  - `mfr2/`: Directory containing the images used in the dataset.
  - `train.txt`, `test.txt`, `validate.txt`: Text files that store the paths of the images and their corresponding labels for training, testing, and validation sets.


## Getting Started

### Prerequisites
To run this project, you need the following dependencies:
- Python >= 3.8
- PyTorch >= 1.9
- torchvision
- einops (used for reshaping tensors)
- numpy
- tqdm (for progress bars)
- matplotlib (for plotting)

You can install the required dependencies using:
```bash
pip install torch torchvision einops numpy tqdm matplotlib
```

### Dataset Preparation

The dataset used in this project is organized as follows:
- `/mfr2`: Contains all the images.
- `train.txt`, `test.txt`, `validate.txt`: Each file contains image paths and their corresponding labels, formatted as: `image_path,label`.

Make sure the dataset is prepared in this structure to ensure successful data loading during training and testing.

### Training the Model

To train the Vision Transformer for mask detection, run the `train.py` script:

```bash
python train.py --data-path ./dataset --batch-size 8 --lr 0.001 --train-steps 20000 --ckpt-path saved_ckpt --cuda
```

Command-line options:
- `--data-path`: Path to the dataset folder.
- `--batch-size`: Batch size for training.
- `--lr`: Initial learning rate.
- `--train-steps`: Total training steps.
- `--ckpt-path`: Path to save model checkpoints.
- `--cuda`: Include this flag if you want to use GPU for training.

The training log will output training accuracy, validation accuracy, and loss after every epoch.

### Evaluating the Model

After training, you can evaluate the model using a confusion matrix and other metrics. Use the `eval_metric.py` script for this purpose.

Run the evaluation with:
```bash
python eval_metric.py --data-path ./dataset --ckpt-path saved_ckpt --cuda
```

This script will output:
- Specificity
- Sensitivity
- Accuracy
- Cohen's Kappa

### Training Logs and Loss Curves

During training, loss curves for both training and validation will be generated and saved as a PDF in the `logs/` directory.

### Testing the Model
To test the saved model, run:
```bash
python test.py
```
This script will use the saved model (`model.pt`) to evaluate the performance on the test set, and it will calculate metrics such as accuracy and generate confusion matrices.

## Project Overview

This project aims to build a **mask detection model** using a Vision Transformer (ViT) architecture. The key objective is to classify images into two categories:
- **Class 1**: Wearing a mask.
- **Class 0**: Not wearing a mask.

We use a Vision Transformer, which is a recent advancement in deep learning that leverages self-attention to achieve remarkable performance on image classification tasks.
