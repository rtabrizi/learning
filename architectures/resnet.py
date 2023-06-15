''' my reimplementation of RESNET 50 guided by Aladdin. With RESNET 50, bottleneck blocks are used
'''
# Imports
import torch
import torch.nn as nn # all nn modules (linear layer, CNNS, loss functions)
import torch.optim as optim # all optimization algorithms like SGD, Adam
import torch.nn.functional as F # all functions that don't have parameters (activation functions, tanh), also included in nn
from torch.utils.data import DataLoader 
import torchvision.datasets as datasets # import pytorch datasets
import torchvision.transforms as transforms # transformations on dataset
import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super().__init__()
        # block 1 example: input channels = 64, output channels = 64 (we ignore the 256 and instead explicilty write that as out_channels * 4)
        # figure 5 of original paper
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

        # for the first layer, we define in_channels to 64, and out_channels to 64
        # when we go to make each layer, every layer but the first has an inputted stride of 2 --> skip connection
        if stride != 1 or in_channels != out_channels * self.expansion:
            # 1x1 convolution
            # we use inputted stride to match feature mapping dimensions
            self.identity_downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_channels * self.expansion))
            
    def forward(self, x):
        identity = x.clone()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        
        return x

class ResNet(nn.Module):
    # layers: how many bottlenecks/blocks  per layer
    def __init__(self, layers, image_channels, num_classes):
        super().__init__()
        self.in_channels = 64

        #initial layers
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #ResNet layers


        #for ResNet 50, input_channels = 64, out_channels is really the 1x1, 64 seen in architecture table
        # we still consider out_channels to be 64 even though last layer has 256 channels (for bottleneck 1)
        self.layer1 = self.__make_layer(layers[0], out_channels=64, stride=1)
        self.layer2 = self.__make_layer(layers[1], out_channels=128, stride=2)
        self.layer3 = self.__make_layer(layers[2], out_channels=256, stride=2)
        self.layer4 = self.__make_layer(layers[3], out_channels=512, stride=2)

        #adaptive --> we define the output size of (1,1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        # expansion of 4
        self.fc = nn.Linear(512*4, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x) 
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def __make_layer(self, num_bottlenecks, out_channels, stride):
        identity_downsample = None
        layers = []
        #changes number of channels only in first bottleneck
        layers.append(Bottleneck(self.in_channels, out_channels, identity_downsample, stride))

        #figure 5 of paper: now that we're past the downsample, the input_channels = 64 * 4 = 256
        self.in_channels = out_channels*4

        for i in range(num_bottlenecks-1):
            layers.append(Bottleneck(self.in_channels, out_channels))

        return nn.Sequential(*layers)
    
def ResNet50(img_channels=3, num_classes=1000):
    return ResNet([3,4,6,3], img_channels, num_classes)

def ResNet101(img_channels=3, num_classes=1000):
    return ResNet([3,8,36,3], img_channels, num_classes)
