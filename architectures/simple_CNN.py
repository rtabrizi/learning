# Imports
import torch
import torch.nn as nn # all nn modules (linear layer, CNNS, loss functions)
import torch.optim as optim # all optimization algorithms like SGD, Adam
import torch.nn.functional as F # all functions that don't have parameters (activation functions, tanh), also included in nn
from torch.utils.data import DataLoader 
import torchvision.datasets as datasets # import pytorch datasets
import torchvision.transforms as transforms # transformations on dataset

# Create Fully Connected Network
class NN(nn.Module):
    def __init__(self, input_size, num_classes): # (28 x 28 = 784 nodes per image)
        super(NN, self).__init__() # calls initialization of parent class nn.Module 
        self.fc1 = nn.Linear(input_size, 50) # fully connected layer 1, to hidden layer with 50 nodes
        self.fc2 = nn.Linear(50, num_classes)
        
    def forward(self, x): # runs on some input x
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

class CNN(nn.Module):

    def __init__(self, in_channels = 1, num_classes = 10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels= in_channels, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1)) # same convolution for input of 28x28
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)) # will divide layer size by 2
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1)) # same convolution for input of 28x28
        self.fc1 = nn.Linear(16*7*7, num_classes) #16 output channels, 28/(2^2) = 14 --> halved twice
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1) # flatten for each batch
        x = self.fc1(x)
        
        return x




model = CNN() # each image is 784 values, 10 possible digits
x = torch.randn(64, 1, 28, 28) #minibatch size of 64 samples, all with 784 values --> it's a 64 x 784 matrix of random numbers between 0 and 1
print(model(x).shape)

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
model = CNN().to(device)

# loss and optimizer
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

# train network
for epoch in range(num_epochs):
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
            
    
