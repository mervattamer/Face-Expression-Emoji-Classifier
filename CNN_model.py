

import torch
import torch.nn as nn
import torchvision.transforms.functional as fn
import torchvision.transforms as transforms


#building cnn

class CNNModel(nn.Module):
     def __init__(self):
          super().__init__()
          self.conv1 = nn.Conv2d(1, 32, kernel_size= (3, 3))
          self.act1 = nn.ReLU()
          self.drop1 = nn.Dropout2d(0.3)

          self.conv2 = nn.Conv2d(32, 64, kernel_size= (3, 3))
          self.act2 = nn.ReLU()
          self.maxpool1 = nn.MaxPool2d(kernel_size= (2, 2))

          self.conv3 = nn.Conv2d(64, 128, kernel_size=(3,3))
          self.act3 = nn.ReLU()
          self.maxpool2 = nn.MaxPool2d(kernel_size = (2,2))
          
          self.conv4 = nn.Conv2d(128, 128, kernel_size=(3,3) )
          self.maxpool3 = nn.MaxPool2d(kernel_size=(2,2))
          self.act4 = nn.ReLU()
          self.drop2 = nn.Dropout2d(0.3)

          self.flat = nn.Flatten()
          self.fc1 = nn.Linear(2048, 1024)
          # added
          self.bn1 = nn.BatchNorm1d(1024)
          self.fcact1 = nn.ReLU()
          self.fc2 = nn.Linear(1024, 512)
          # added
          self.bn2 = nn.BatchNorm1d(512)
          self.fcact2 = nn.ReLU()
          self.fc3 = nn.Linear(512, 10)

     def forward(self, x):
          x = self.act1(self.conv1(x))
          x = self.drop1(x)
          x = self.act2(self.conv2(x))
          x = self.maxpool1(x)
          x = self.act3(self.conv3(x))
          x = self.maxpool2(x)
          x = self.act4(self.conv4(x))
          x = self.maxpool3(x)
          x = self.drop2(x)
          x = self.flat(x)
          x = self.fc1(x)
          x = self.bn1(x)
          x = self.fcact1(x)
          x = self.fc2(x)
          x = self.bn2(x)
          x = self.fcact2(x)
          x = self.fc3(x)
          
          return x 
     def predict(self,x):
          x = transforms.ToTensor()(x)
          x = x.unsqueeze(0)
          #print(x.shape)  
          self.eval()
          with torch.no_grad():
             x = self.forward(x)
          self.train()
          return x
