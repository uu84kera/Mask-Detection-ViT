import os
from datetime import datetime
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from vit_pytorch.simple_vit import SimpleViT
import numpy as np
import random
import matplotlib.pyplot as plt

from dataset_loader import SimpleVitDataset  # Assuming your custom dataset class
from torch.optim import SGD
from torchvision import transforms

"""
This script is used to train the network using ViT.
"""

# Define command-line arguments for configuration
parser = argparse.ArgumentParser()
parser.add_argument('--data-path', type=str, default='./dataset', help='Dataset path')
parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
parser.add_argument('--train-steps', type=int, default=20000, help='Total training steps')
parser.add_argument('--warmup-steps', type=int, default=500, help='Learning rate warm-up steps')
parser.add_argument('--ckpt-path', type=str, default='saved_ckpt', help='Checkpoint path')
parser.add_argument('--logs-path', type=str, default='./logs', help='Logs folder path')
parser.add_argument('--cuda', action='store_true', help='If use cuda')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

# Accuracy computation function
def accuracy(output, target, mode='train'):
    """Computes the accuracy for the specified values of k"""
    batch_size = target.size(0)
    pred = torch.argmax(output, dim=1)
    correct = pred.eq(target)
    correct = correct.float().sum()
    if mode == 'train':
        correct = correct / batch_size
    return correct

# Set the random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # Ensures reproducibility even when using CUDA
    torch.backends.cudnn.benchmark = False  # Disables cuDNN auto-tuner for deterministic behavior

if __name__ == '__main__':
    args = parser.parse_args()

    # Set the random seed for reproducibility
    set_seed(args.seed)

    # Check if CUDA is available
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    # Create the checkpoint directory if it does not exist
    if not os.path.exists(args.ckpt_path):
        os.makedirs(args.ckpt_path, exist_ok=True)
    
    # Create the logs directory if it does not exist
    if not os.path.exists(args.logs_path):
        os.makedirs(args.logs_path, exist_ok=True)

    # Set up the model (SimpleViT)
    model = SimpleViT(
        image_size=256,
        patch_size=32,
        num_classes=2,  # Modify this to match your dataset
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048
    )
    model.to(device)  # Move model to device

    # Load the training dataset
    train_set = SimpleVitDataset(args.data_path, os.path.join(args.data_path, 'train.txt'), image_size=(256, 256), mode='train')
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Load the validation dataset
    validate_set = SimpleVitDataset(args.data_path, os.path.join(args.data_path, 'validate.txt'), image_size=(256, 256), mode='validate')
    validate_loader = DataLoader(dataset=validate_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Load the test dataset
    test_set = SimpleVitDataset(args.data_path, os.path.join(args.data_path, 'test.txt'), image_size=(256, 256), mode='test')
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=args.lr, weight_decay=1e-4, momentum=0.9)

    # Define learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=args.lr,
        pct_start=args.warmup_steps / args.train_steps,
        total_steps=args.train_steps
    )

    # Training loop
    best_acc = 0
    epochs = 50  # Train for 50 epochs
    train_losses = []
    validate_losses = []

    for epoch_id in range(epochs):
        model.train()
        epoch_train_loss = 0

        for batch_idx, (batch_data, batch_target) in enumerate(train_loader):
            batch_data, batch_target = batch_data.to(device), batch_target.to(device)

            # Forward pass
            batch_pred = model(batch_data)
            loss = criterion(batch_pred, batch_target)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            # Calculate accuracy
            train_acc = accuracy(batch_pred, batch_target)
            epoch_train_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f'[{datetime.now()}] Epoch {epoch_id+1}/{epochs}, Step {batch_idx+1}/{len(train_loader)}, '
                      f'Train Accuracy: {train_acc.item():.3f}, Loss: {loss.item():.4f}, Best Accuracy: {best_acc:.3f}')
        
        # Store epoch training loss
        train_losses.append(epoch_train_loss / len(train_loader))

        # Validate the model
        if epoch_id % 1 == 0:
            model.eval()
            epoch_val_loss = 0
            validate_acc = 0
            with torch.no_grad():
                for batch_idx, (batch_data, batch_target) in enumerate(validate_loader):
                    batch_data, batch_target = batch_data.to(device), batch_target.to(device)
                    batch_pred = model(batch_data)
                    loss = criterion(batch_pred, batch_target)
                    acc_tmp = accuracy(batch_pred, batch_target, mode='validate')
                    epoch_val_loss += loss.item()
                    validate_acc += acc_tmp.item()

            validate_acc /= len(validate_loader)
            validate_losses.append(epoch_val_loss / len(validate_loader))
            print(f'[{datetime.now()}] Epoch {epoch_id+1}/{epochs}, Validation Accuracy: {validate_acc:.3f}')

            # Save the model if the accuracy is improved
            if validate_acc > best_acc:
                best_acc = validate_acc
                torch.save(model.state_dict(), os.path.join(args.ckpt_path, 'model.pt'))
                print(f'[{datetime.now()}] Model saved with accuracy: {best_acc:.3f}')

    # Test the model (optional)
    model.eval()
    test_acc = 0
    with torch.no_grad():
        for batch_idx, (batch_data, batch_target) in enumerate(test_loader):
            batch_data, batch_target = batch_data.to(device), batch_target.to(device)
            batch_pred = model(batch_data)
            acc_tmp = accuracy(batch_pred, batch_target, mode='test')
            test_acc += acc_tmp.item()

    test_acc /= len(test_loader)
    print(f'[{datetime.now()}] Test Accuracy: {test_acc:.3f}')

    # Plot the training and validation loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(range(epochs), train_losses, label='Training Loss', color='blue')
    plt.plot(range(epochs), validate_losses, label='Validation Loss', color='orange')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Save the plot as a PDF file in the logs directory
    loss_curve_path = os.path.join(args.logs_path, 'loss_curve.pdf')
    plt.savefig(loss_curve_path, format='pdf')

    # Show the plot
    plt.show()
