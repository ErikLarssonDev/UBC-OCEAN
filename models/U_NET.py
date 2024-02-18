import torch 
import torch.nn as nn
import pandas as pd
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os

sys.path.append("../encoder")
from encoder_dataset import UBCDataset 

config = {'DEVICE': "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")}

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False), # Same convolution, input == output dim. No bias due to batchnorm
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)
    
class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[16, 32, 64, 128, 256]):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of U-NET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        # Up part of U-NET
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck= DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1] # Reversing the list

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            if x.shape != skip_connection.shape: # If we do inputs which are not divisible by 16 we need this
                x = TF.resize(x, size=skip_connection.shape[2:]) # Just taking out height and width, not batch size and channels
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        
        return self.final_conv(x)
    
class UNET_no_skip(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[16, 32, 64, 128, 256]):
        super(UNET_no_skip, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of U-NET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        # Up part of U-NET
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck= DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            # skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        # skip_connections = skip_connections[::-1] # Reversing the list

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            # skip_connection = skip_connections[idx//2]
            # if x.shape != skip_connection.shape: # If we do inputs which are not divisible by 16 we need this
            #     x = TF.resize(x, size=skip_connection.shape[2:]) # Just taking out height and width, not batch size and channels
            # concat_skip = torch.cat((skip_connection, x), dim=1)
            # x = self.ups[idx+1](x)
        
        return self.final_conv(x)

# def test_with_real_batch():
#     traindf = pd.read_csv("D:\\OvarianCancerData\\processedThumbnails\\train_tiles.csv")
#     train_dataset = UBCDataset(traindf, transforms=config['DATA_TRANSFORMS']["train"])
#     dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
#     # Get one batch
#     for batch, (X, y) in enumerate(dataloader):
#         # x = torch.randn((256,256,3,3))
#         X, y = X.to(config['DEVICE']), y.to(config['DEVICE'])
#         model = UNET(in_channels=3, out_channels=3).to(config['DEVICE'])
#         preds = model(X)
#         print(preds.shape)
#         print(X.shape)
#         print("Success!")
#     assert preds.shape == X.shape

def test():
    x = torch.randn((3, 3, 1024, 1024)).to(config['DEVICE'])
    model = UNET_no_skip().to(config['DEVICE'])
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    print("Success!")
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()