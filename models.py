## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # convolution layers
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2) #112*112*32 output
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) #56*56*128 output
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1) #28*28*256 output
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1) #14*14*256 output
        self.conv5 = nn.Conv2d(256, 256, 3, padding=1) #7*7*256 output
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1) #3*3*256 output
        
        # maxpooling layer
        self.pool = nn.MaxPool2d(2, 2) #After Appling pooling layer it gives: 110*110*32 output
        
        #batch normalization
        self.batchn1 = nn.BatchNorm2d(num_features= 32, eps=1e-05)
        self.batchn2 = nn.BatchNorm2d(num_features= 64, eps=1e-05)
        self.batchn3 = nn.BatchNorm2d(num_features= 128, eps=1e-05)
        self.batchn4 = nn.BatchNorm2d(num_features= 256, eps=1e-05)
        self.batchn5 = nn.BatchNorm2d(num_features= 512, eps=1e-05)
        
        #linear layers
        self.fc1 = nn.Linear(3*3*256,2048)
        self.fc2 = nn.Linear(2048,512)
        self.fc3 = nn.Linear(512, 136)
        
        # dropout with p=0.4
        self.dropout = nn.Dropout(p=0.30)
        
   
        
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool(self.batchn1(F.relu(self.conv1(x))))
        x = self.pool(self.batchn2(F.relu(self.conv2(x))))
        x = self.pool(self.batchn3(F.relu(self.conv3(x))))
        x = self.pool(self.batchn4(F.relu(self.conv4(x))))
        x = self.pool(self.batchn4(F.relu(self.conv5(x))))
        x = self.pool(self.batchn4(F.relu(self.conv6(x))))
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x