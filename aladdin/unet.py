from turtle import pd
import torch
import torch.nn as nn
import torchvision.transforms.functional
# padding=0 in original paper
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        # self.pool = nn.MaxPool2d(kernel_size=(2,2))

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

# encoding, i.e. the descent to the bottom of the U
class DownSample(nn.Module):
    def __init__(self):
        super().__init__()

        self.pool = nn.MaxPool2d(kernel_size=(2,2)) 
    
    def forward(self, x):
        x = self.pool(x)
        return x

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.upConv = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.upConv(x)
        return x

class CropAndConcat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, contracting_x):
        contracting_x = torchvision.transforms.functional.center_crop(contracting_x, [x.shape[2], x.shape[3]])
        x = torch.cat([x, contracting_x], dim=1)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.down_conv = nn.ModuleList([DoubleConv(in_channels, out_channels) for in_channels, out_channels in [(in_channels, 64), (64, 128), (128, 256), (256, 512)]])
        self.down_sample = nn.ModuleList([DownSample() for layer in range(4)])
        self.middle_conv = DoubleConv(512, 1024)
        self.up_sample = nn.ModuleList([UpSample(in_channels, out_channels) for in_channels, out_channels in [(1024, 512), (512, 256), (256, 128), (128, 64)]])
        self.up_conv = nn.ModuleList([DoubleConv(in_channels, out_channels) for in_channels, out_channels in [(1024, 512), (512, 256), (256, 128), (128, 64)]])
        self.concat = nn.ModuleList([CropAndConcat() for layer in range(4)])
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        pass_through = []

        for layer in range(len(self.down_conv)):
            x = self.down_conv[layer](x)
            pass_through.append(x)
            x = self.down_sample[layer](x)
        x = self.middle_conv(x)
        for layer in range(len(self.up_conv)):
            x = self.up_sample[layer](x)
            x = self.concat[layer](x, pass_through.pop())
            x = self.up_conv[layer](x)
        x = self.final_conv(x)

        return x

x = torch.randn(64, 1, 32, 32)
model = UNet(1, 10)




