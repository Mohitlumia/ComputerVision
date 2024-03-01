
##########################################################
# download the Training and Validation MNIST digit images


import torchvision.datasets as dsets        # Allows us to get the digit dataset
import torchvision.transforms as transforms # Allows us to transform data


train_dataset = dsets.MNIST(root='./Image Classifier/MNIST', train=True, download=True, transform=transforms.ToTensor())
print("Print the training dataset:\n ", train_dataset)

validation_dataset = dsets.MNIST(root='./Image Classifier/MNIST', download=True, transform=transforms.ToTensor())
print("Print the validation dataset:\n ", validation_dataset)

print(train_dataset[3])     # print the image at index 3
print(train_dataset[3][1])  # print the lable of the image at index 3


##########################################################
# Create a Softmax Classifier using PyTorch


import torch.nn as nn   # PyTorch Neural Network


# Define softmax classifier class
# Inherits nn.Module which is the base class for all neural networks
class SoftMax(nn.Module):
    
    # Constructor
    def __init__(self, input_size, output_size):
        super(SoftMax, self).__init__()
        # Creates a layer of given input size and output size
        self.linear = nn.Linear(input_size, output_size)
        
    # Prediction
    def forward(self, x):
        # Runs the x value through the single layers defined above
        z = self.linear(x)
        return z
