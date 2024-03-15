

import torchvision.datasets as dsets        # Allows us to get the digit dataset
import torchvision.transforms as transforms # Allows us to transform data
import torch.nn as nn                       # PyTorch Neural Network
import torch.optim, torch.utils
from matplotlib import pyplot as plt

##########################################################
# download the Training and Validation MNIST digit images

train_dataset = dsets.MNIST(root='./Data/', train=True, download=False, transform=transforms.ToTensor())

validation_dataset = dsets.MNIST(root='./Data/', download=False, transform=transforms.ToTensor())

##########################################################
# Create a Softmax Classifier using PyTorch

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


# Set input size and output size
input_dim = 28 * 28     #image size
output_dim = 10         #number of class of images

# Create the model
# Input dim is 28*28 which is the image converted to a tensor
# Output dim is 10 because there are 10 possible digits the image can be
model = SoftMax(input_dim, output_dim)

#################################################################################
# Define the learning rate, optimizer, criterion, and data loader

learning_rate = 0.1

# The optimizer will updates the model parameters using the learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# The criterion will measure the loss between the prediction and actual label values
# This is where the SoftMax occurs, it is built into the Criterion Cross Entropy Loss
criterion = nn.CrossEntropyLoss()

# Created a training data loader so we can set the batch size
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100)

# Created a validation data loader so we can set the batch size
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5000)

#########################################################################################
# Number of times we train our model useing the training data
n_epochs = 10
# Lists to keep track of loss and accuracy
loss_list = []
accuracy_list = []
# Size of the validation data
N_test = len(validation_dataset)

# Function to train the model based on number of epochs
def train_model(n_epochs):
    # Loops n_epochs times
    for epoch in range(n_epochs):
        # For each batch in the train loader
        for x, y in train_loader:
            # Resets the calculated gradient value, this must be done each time as it accumulates if we do not reset
            optimizer.zero_grad()
            # flat the image
            flat_x = x.view(-1, 28 * 28)
            # Makes a prediction based on the image tensor
            z = model(flat_x)
            # Calculates loss between the model output and actual class
            loss = criterion(z, y)
             # Calculates the gradient value with respect to each weight and bias
            loss.backward()
            # Updates the weight and bias according to calculated gradient value
            optimizer.step()
        
        # Each epoch we check how the model performs with data it has not seen which is the validation data, we are not training here
        correct = 0
        # For each batch in the validation loader
        for x_test, y_test in validation_loader:
            # flat the test data
            flat_x_test = x_test.view(-1, 28 * 28)
            # Makes prediction based on image tensor
            z = model(flat_x_test)
            # Finds the class with the higest output
            _, yhat = torch.max(z.data, 1)
            # Checks if the prediction matches the actual class and increments correct if it does
            correct += (yhat == y_test).sum().item()
        # Calculates the accuracy by dividing correct by size of validation dataset
        accuracy = correct / N_test
        # Keeps track loss
        loss_list.append(loss.data)
        # Keeps track of the accuracy
        accuracy_list.append(accuracy)

# Function call
train_model(n_epochs)

####################################################################
# Plot the loss and accuracy

fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.plot(loss_list,color=color)
ax1.set_xlabel('epoch',color=color)
ax1.set_ylabel('total loss',color=color)
ax1.tick_params(axis='y', color=color)
    
ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('accuracy', color=color)  
ax2.plot( accuracy_list, color=color)
ax2.tick_params(axis='y', color=color)
fig.tight_layout()

###################################################################
# Save the plot as an image

from PIL import Image

# Convert Matplotlib figure to PIL Image
fig.canvas.draw()
buf = fig.canvas.tostring_rgb()
width, height = fig.canvas.get_width_height()
pil_image = Image.frombytes("RGB", (width, height), buf)
 
# Save PIL Image as PNG
pil_image.save('Image Classifier/Softmax Classifier/total loss and accuracy.png')
