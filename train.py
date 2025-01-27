from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from tqdm.notebook import tqdm
import wandb
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, class_names):
    # Normalize the confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    df_cm = pd.DataFrame(cm_normalized, index=class_names, columns=class_names)
    plt.figure(figsize=(10,7))
    sns.heatmap(df_cm, annot=True)
    plt.show()

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for batch_idx, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}/{len(loader)}: Loss: {loss.item():.4f}, Accuracy: {100.0 * total_correct / total_samples:.2f}%")

    average_loss = total_loss / len(loader)
    accuracy = 100.0 * total_correct / total_samples
    return average_loss, accuracy

def validate_epoch(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    average_loss = total_loss / len(loader)
    accuracy = 100.0 * total_correct / total_samples
    uar = (confusion_matrix.diag() / confusion_matrix.sum(1)).mean().item() * 100
    return average_loss, accuracy, uar, confusion_matrix.cpu().numpy()

def train_model(model, train_loader, val_loader, optimizer, criterion, class_names, num_epochs, project_name, ident_str=None, class_weights_tensor=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Adjust criterion for class weights if provided
    if class_weights_tensor is not None:
        class_weights_tensor = class_weights_tensor.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    ident_str = ident_str or datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{model.__class__.__name__}_{ident_str}"
    wandb_run = wandb.init(project=project_name, name=exp_name)

    for epoch in tqdm(range(num_epochs), desc='Epochs', total=num_epochs):
        train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy, uar, confusion_mat = validate_epoch(model, val_loader, criterion, device, len(class_names))

        # Logging to wandb
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'uar': uar
        })

        # Plot the confusion matrix at the last epoch
        if epoch == num_epochs - 1:
            plot_confusion_matrix(confusion_mat, class_names)

    wandb_run.finish()

# Assume the rest of your code for dataset preparation and model instantiation goes here.

# Example usage:
# train_model(model, train_loader, val_loader, optimizer, criterion, class_names, NUM_EPOCHS, PROJECT_NAME, ident_str=IDENT_STR, class_weights_tensor=class_weights_tensor)
