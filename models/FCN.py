import torchvision
import torch 
import torch.nn as nn
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.pretrained_model = torchvision.models.segmentation.fcn_resnet101(weights=torchvision.models.segmentation.FCN_ResNet101_Weights.DEFAULT)
        self.pretrained_model.classifier[4] = nn.Conv2d(512, 3, kernel_size=(1, 1), stride=(1, 1))
        self.pretrained_model.aux_classifier[4] = nn.Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        return self.pretrained_model(x)

def test():
    x = torch.randn((3, 3, 256, 256)).to(DEVICE)
    model = FCN().to(DEVICE)
    preds = model(x)["out"]
    print(preds.shape)
    print(x.shape)
    print("Success!")
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()