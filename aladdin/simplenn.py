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
    
model = NN(784, 10) # each image is 784 values, 10 possible digits
x = torch.randn(64, 784) #minibatch size of 64 samples, all with 784 values --> it's a 64 x 784 matrix of random numbers between 0 and 1

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
input_size = 784
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
model = NN(input_size = input_size, num_classes = num_classes).to(device)

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
        # -1 flattens everything else into single dimension
        images = images.resjhape(images.shape[0], -1) 
        
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
            images = images.reshape(x.shape[0], -1)
            
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
            
    
