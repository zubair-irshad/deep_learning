import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, im_size, hidden_dim,hidden_dim2, kernel_size, n_classes):
        '''
        Create components of a CNN classifier and initialize their weights.

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            kernel_size (int): Width and height of (square) convolution filters
            n_classes (int): Number of classes to score
        '''
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3,16,3,padding=1)
        (C,H,W) = im_size
        self.conv2 = nn.Conv2d(16,32,3,padding=1)
        self.conv3 = nn.Conv2d(32,64,3,padding=1)
        self.conv4 = nn.Conv2d(64,128,3,padding=1)
        
        self.pool = nn.MaxPool2d(2,2)
        #To calcualte dimension of fully connected, we flatted output of last pooling layer so it has dimension (_*64(depth))
        #_ is Width: how muchbox has shruken. 3 pooling layers mean height/2^3
        
        hout_size = H/(2**4)
        print(H)
        print(hout_size)
        
        self.fc1  = nn.Linear(hout_size*hout_size*128,hidden_dim)
        self.fc2  = nn.Linear(hidden_dim,hidden_dim2)
        self.fc3  = nn.Linear(hidden_dim2,n_classes)
#         self.out  = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=0.25)
        
        

    def forward(self, images):
        '''
        Take a batch of images and run them through the CNN to
        produce a score for each class.

        Arguments:
            images (Variable): A tensor of size (N, C, H, W) where
                N is the batch size
                C is the number of channels
                H is the image height
                W is the image width

        Returns:
            A torch Variable of size (N, n_classes) specifying the score
            for each example and category.
        '''
        x = self.pool(F.relu(self.conv1(images)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        
        #FLatten the output to a vector
        x = x.view(x.shape[0],-1)
        x=self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        scores = x
#         scores = self.out(x)
        #############################################################################
        # TODO: Implement the forward pass. This should take few lines of code.
        #############################################################################
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return scores

