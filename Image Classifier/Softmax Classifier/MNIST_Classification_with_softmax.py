
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

train_dataset = dsets.MNIST(root='Image Classifier/MNIST', train=True, download=True, transform=transforms.ToTensor())
validation_dataset = dsets.MNIST(root='Image Classifier/MNIST', download=True, transform=transforms.ToTensor())

# Set input size and output size
input_dim = 28 * 28
output_dim = 10

# Create the model
# Input dim is 28*28 which is the image converted to a tensor
# Output dim is 10 because there are 10 possible digits the image can be
model = SoftMax(input_dim, output_dim)

# First we get the X value of the first image
X = train_dataset[0][0]

# We can see the shape is 1 by 28 by 28, we need it to be flattened to 1 by 28 * 28 (784)
X = X.view(-1, 28*28)

# Now we can make a prediction, each class has a value, and the higher it is the more confident the model is that it is that digit
# compare model output one first image with its label
import torch

model_output = model(X)
actual = torch.tensor([train_dataset[0][1]])

print("Output: ", model_output)
print("Actual:", actual)

# Define the learning rate, optimizer, criterion, and data loader

learning_rate = 0.1

# The optimizer will updates the model parameters using the learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# The criterion will measure the loss between the prediction and actual label values
# This is where the SoftMax occurs, it is built into the Criterion Cross Entropy Loss
criterion = nn.CrossEntropyLoss()
print(criterion(model_output, actual))

# Created a training data loader so we can set the batch size
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100)

# Created a validation data loader so we can set the batch size
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5000)
