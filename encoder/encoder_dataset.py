import torch
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
import os
import numpy as np
import cv2

class UBCDataset(Dataset):
    def __init__(self, dataset_df, transforms=None):
        self.file_paths = dataset_df['file_path']
        self.transforms = transforms
        self.df = dataset_df
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        img_path = self.file_paths[index]
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error loading image: {img_path}")
        
        if self.transforms:
            img = self.transforms(image=img)["image"]
            
        return img, img

