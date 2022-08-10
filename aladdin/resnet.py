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

class Block(nn.Module):
    # identity downsample for moments where there's a dotted line when looking at paper's graph
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super().__init__()
        # block 1 example: input channels = 64, output channels = 64 (we ignore the 256 and instead explicilty write that as out_channels * 4)
        # figure 5 of original paper
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
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
    # layers will be list of how many times per block [3, 4, 6, 3]
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
        # we still consider out_channels to be 64 even though last layer has 256 channels (for block 1)
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

    def __make_layer(self, num_residual_blocks, out_channels, stride):
        identity_downsample = None
        layers = []

        

        #changes number of channels only in first block
        layers.append(Block(self.in_channels, out_channels, identity_downsample, stride))

        #figure 5 of paper: now that we're past the downsample, the input_channels = 64 * 4 = 256
        self.in_channels = out_channels*4

        for i in range(num_residual_blocks-1):
            layers.append(Block(self.in_channels, out_channels))

        return nn.Sequential(*layers)
    
def ResNet50(img_channels=3, num_classes=1000):
    return ResNet([3,4,6,3], img_channels, num_classes)

def ResNet101(img_channels=3, num_classes=1000):
    return ResNet([3,8,36,3], img_channels, num_classes)

model = ResNet50(img_channels=1, num_classes=10) # each image is 784 values, 10 possible digits

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
in_channels = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1

# Load Data
train_dataset = datasets.MNIST(root = 'dataset/', train = True, download = True, transform = transforms.ToTensor()) # root is where dataset is saved
# transforms data (default is numpy array) to tensors
train_loader = DataLoader(dataset=train_dataset, batch_size = batch_size, shuffle=True) # shuffle --> shuffles batches each epoch for variety
test_dataset = datasets.MNIST(root = 'dataset/', train = False, transform = transforms.ToTensor()) 
test_loader = DataLoader(dataset=test_dataset, batch_size = batch_size, shuffle=True) 

# initialize network

# loss and optimizer
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

# train network
print("NUM EPOCHS: %s" % num_epochs)
print("NUM BATCHES: %s" % len(train_loader))
for epoch in range(num_epochs):
    print("EPOCH: %s" % str(epoch + 1))
    for batch_idx, (images, labels) in enumerate(train_loader): # enumerate shows batch index, data is image, target is correct label
        print("batch number: %s" % str(batch_idx))
        # get data to cuda if possible
        images = images.to(device=device)
        labels = labels.to(device=device)
        #x.shape = (64,1,28,28) batch size of 64, 1 color channel, 28x28 pixel
        
        # forward
        
        logits = model(images)
        
        #calculate cost
        J = loss(logits, labels)
        
        #backward
        #set all gradiants to zero for each batch so it isn't stored from previous forward props
        optimizer.zero_grad()
        
        J.backward()
        
        
        # gradient descent or adam step
        optimizer.step()
        
        

# check accuracy
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("checking accuracy on training data")
    else:
        print("checking accuracy on testing data")
    num_correct = 0
    num_samples = 0
    
    # set model to evaluation mode
    model.eval()
    
    # when checking accuracy, don't calculate the gradients when checking accuracy
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device=device)
            labels = labels.to(device=device)
    
            
            logits = model(images)
            # max of second dimension (value from 0 to 9 for certain digit) 64 x 10
            _, predictions = logits.max(1)
            num_correct += (predictions == labels).sum()
            # batch size = 64 = number of samples
            num_samples += predictions.size(0)
            
        #converts tensors to floats with 2 decimals    
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}') 
        
    model.train()
check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
            
    
