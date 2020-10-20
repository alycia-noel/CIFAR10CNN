"""
Network for CIFAR-10 classification

Created on Sun Oct 11 11:39:05 2020

@author: ancarey
"""
import torch.nn as nn
import torch.nn.functional as F 

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        
        #Conv2D = (in channels, out channels, kernel size, padding)
        self.conv_layer = nn.Sequential( 
            
            #Layer 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), #kernel_size, stride,
            
            #Layer 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.8),
            
            #Layer 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.8)
        
            # #Layer 4
            # nn.Conv2d(256, 512, 3, padding=1),
            # nn.BatchNorm2d(512),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, 3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(2,2)
        )
        
        self.fc_layer = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(.7),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(.7),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.7),
            nn.Linear(512, 10)
            #nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1) #flatten for fc layers
        x = self.fc_layer(x)
        
        return x
        
