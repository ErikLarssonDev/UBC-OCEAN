from collections import OrderedDict
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torchvision.utils import make_grid
import sys
import os
from encoder_dataset import UBCDataset
from sklearn.model_selection import train_test_split
from config import config
from torch.utils.data import DataLoader

traindf = pd.read_csv(os.path.join(config['INPUT_DIR'], 'train_tiles_updated.csv'))

train_data, val_data = train_test_split(traindf, test_size=config['SPLIT_RATIO'], random_state=config['SEED'])
train_data = train_data.reset_index(drop=True)
val_data = val_data.reset_index(drop=True)

train_dataset = UBCDataset(train_data, transforms=config['DATA_TRANSFORMS']["train"])
train_dataloader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'], shuffle=True, pin_memory=True)

val_dataset = UBCDataset(val_data, transforms=config['DATA_TRANSFORMS']["valid"]) 
val_dataloader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'], 
                            num_workers=config['NUM_WORKERS'], shuffle=False, pin_memory=True)
# model = UNET().to(config['DEVICE'])
# model = UNET_no_skip().to(config['DEVICE'])
# model = FCN().to(config['DEVICE'])
model = torch.load(f"../models/saved/{config['MODEL_NAME']}.pth")
# model = torch.load('../models/saved/FCN_50e_split.pth')

# Set your model to evaluation mode
model.eval()

# Initialize variables to store images and labels
images_list = []
labels_list = []

# Iterate over val_dataloader to get predictions
with torch.no_grad():
    for images, labels in val_dataloader:
        images, labels = images.to(config['DEVICE']), labels.to(config['DEVICE'])

        # Forward pass to get predictions
        predictions = model(images) 
        if isinstance(predictions, OrderedDict):
            predictions = predictions["out"]

        # Append images and labels to the lists
        images_list.append(images.cpu())
        labels_list.append(predictions.cpu())

# Concatenate batches into single tensors
all_images = torch.cat(images_list)
all_labels = torch.cat(labels_list)

# Select 10 random samples
indices = np.random.choice(len(all_images), size=10, replace=False)
selected_images = all_images[indices]
selected_labels = all_labels[indices]

# Display ground truth and predicted data
fig, axs = plt.subplots(3, 2, figsize=(8, 20))
for i in range(3):
    # Display ground truth
    axs[i, 0].imshow(np.transpose(selected_images[i], (1, 2, 0)))
    axs[i, 0].set_title("Ground Truth")
    axs[i, 0].axis("off")

    # Display predicted data
    axs[i, 1].imshow(np.transpose(selected_labels[i], (1, 2, 0)))
    axs[i, 1].set_title("Predicted")
    axs[i, 1].axis("off")

plt.show()