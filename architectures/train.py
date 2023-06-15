import torch
import torch.nn as nn # all nn modules (linear layer, CNNS, loss functions)
import torch.optim as optim # all optimization algorithms like SGD, Adam
import torch.nn.functional as F # all functions that don't have parameters (activation functions, tanh), also included in nn
from torch.utils.data import DataLoader 
import torchvision.datasets as datasets # import pytorch datasets
import torchvision.transforms as transforms # transformations on dataset
import torch
import torch.nn as nn
from aladdin.resnet import ResNet50
from unet import UNet
import wandb
#  wandb.login()


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
# config = dict(epochs = 1, classes=10, batch_size=64, learning_rate=0.001, dataset="MNIST")
# def model_pipeline(hyperparameters):

    
in_channels = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1

# model = ResNet50(img_channels=in_channels, num_classes=num_classes) # each image is 784 values, 10 possible digits
model = UNet(in_channels=in_channels, num_classes=num_classes)

# Load Data
train_dataset = datasets.MNIST(root = 'dataset/', train = True, download = True, transform = transforms.ToTensor()) # root is where dataset is saved
# transforms data (default is numpy array) to tensors
train_loader = DataLoader(dataset=train_dataset, batch_size = batch_size, shuffle=True) # shuffle --> shuffles batches each epoch for variety
test_dataset = datasets.MNIST(root = 'dataset/', train = False, transform = transforms.ToTensor()) 
test_loader = DataLoader(dataset=test_dataset, batch_size = batch_size, shuffle=True) 

images, labels = next(iter(train_loader))
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
            
    
