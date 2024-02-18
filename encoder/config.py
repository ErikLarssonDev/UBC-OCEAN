import torch
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2

import sys
sys.path.append("../models")
from U_NET import UNET, UNET_no_skip
from FCN import FCN
from dotenv import dotenv_values
env_config = dotenv_values("../.env")


IMAGE_SIZE = 1024
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

config = {
    'MODEL': UNET().to(DEVICE),
    'MODEL_NAME': "UNET_80e_split_1024_tiles",
    'WANDB': False,
    'INPUT_DIR': env_config["DATA_DIR"],
    'SPLIT_RATIO': 0.2,# Validation split ratio
    'EPOCHS': 80,
    'BATCH_SIZE': 4,
    'DEVICE': DEVICE,
    'NUM_WORKERS': 0,
    'SEED': 42,
    'IMAGE_SIZE': IMAGE_SIZE,
    'DATA_TRANSFORMS': {
        "valid": A.Compose([
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
            ToTensorV2()], p=1.),
        "train": A.Compose([
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.75),
            A.OneOf([
                A.GaussNoise(var_limit=[10, 50]),
                A.GaussianBlur(),
                A.MotionBlur(),
            ], p=0.4),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
            A.CoarseDropout(max_holes=1, max_width=int(512 * 0.3),
                            max_height=int(512 * 0.3), mask_fill_value=0, p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
            ToTensorV2()], p=1.)
    },
}