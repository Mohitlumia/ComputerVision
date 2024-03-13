
# PyTorch Library
import torch 
# PyTorch Neural Network
import torch.nn as nn
# Allows us to transform tensors
import torchvision.transforms as transforms
# Allows us to download datasets
import torchvision.datasets as dsets
# Allows us to use activation functions
import torch.nn.functional as F
# Used to graph data and loss curves
import matplotlib.pylab as plt
# Allows us to use arrays to manipulate and store data
import numpy as np
# Setting the seed will allow us to control randomness and give us reproducibility
torch.manual_seed(2)


#################################################################################
# Create the model class using Sigmoid as the activation function

class Net(nn.Module):
    
    # Constructor
    def __init__(self, D_in, H1, H2, D_out):
        # D_in is the input size of the first layer (size of input layer)
        # H1 is the output size of the first layer and input size of the second layer (size of first hidden layer)
        # H2 is the outpout size of the second layer and the input size of the third layer (size of second hidden layer)
        # D_out is the output size of the third layer (size of output layer)
        super(Net, self).__init__()
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, D_out)
    
    # Prediction
    def forward(self,x):
        # Puts x through the first layers then the sigmoid function
        x = torch.sigmoid(self.linear1(x)) 
        # Puts results of previous line through second layer then sigmoid function
        x = torch.sigmoid(self.linear2(x))
        # Puts result of previous line through third layer
        x = self.linear3(x)
        return x
    

#################################################################################
# Create the model class using Relu as the activation function

class NetRelu(nn.Module):
    
    # Constructor
    def __init__(self, D_in, H1, H2, D_out):
        # D_in is the input size of the first layer (size of input layer)
        # H1 is the output size of the first layer and input size of the second layer (size of first hidden layer)
        # H2 is the outpout size of the second layer and the input size of the third layer (size of second hidden layer)
        # D_out is the output size of the third layer (size of output layer)
        super(NetRelu, self).__init__()
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, D_out)
    
    # Prediction
    def forward(self, x):
        # Puts x through the first layers then the relu function
        x = torch.relu(self.linear1(x))  
        # Puts results of previous line through second layer then relu function
        x = torch.relu(self.linear2(x))
        # Puts result of previous line through third layer
        x = self.linear3(x)
        return x