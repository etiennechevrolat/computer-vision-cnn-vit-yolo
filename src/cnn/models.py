import torch
import torch.nn as nn

### CNN 1
class CNN1(nn.Module):  
    def __init__(self): 
        super().__init__() 
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, padding=2, stride=1)
        # 32x32 -> pool1 -> 16x16 -> pool2 -> 8x8. Conv3 garde 8x8.
        self.fc1 = nn.Linear(in_features=16 * 8 * 8, out_features=10) 

    def forward(self, x): 
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.relu(self.conv3(x)) # Ajouté pour cohérence
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        return x

### Influence of the depth of the filters

# CNN 2
class CNN2(nn.Module):
    def __init__(self): 
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=2, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=5, padding=2, stride=1)
        # 32x32 -> 16x16 -> 8x8
        self.fc1 = nn.Linear(in_features=8 * 8 * 8, out_features=10) 

    def forward(self, x): 
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.relu(self.conv3(x))
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        return x

### Filters of smaller size 

# CNN 3
class CNN3(nn.Module):
    def __init__(self): 
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=2, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) 
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) 
        self.conv3 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1, stride=1)
        # 32x32 -> 16x16 -> 8x8
        self.fc1 = nn.Linear(in_features=8 * 8 * 8, out_features=10) 

    def forward(self, x): 
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.relu(self.conv3(x))
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        return x
### CNN 4 - One conv layer 

class CNN4(nn.Module):
    def __init__(self):
        super().__init__() 
        self.conv = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=5, padding=2)
        # Pas de pooling, donc on reste en 32x32
        self.fc = nn.Linear(in_features=4 * 32 * 32, out_features=10) 

    def forward(self, x): 
        x = torch.relu(self.conv(x))
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

### Classic model : LeNet
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0)
        self.pool4 = nn.MaxPool2d(kernel_size=5, stride=2, padding=2)
        # 32x32 -> pool2 -> 16x16 -> conv3 -> 12x12 -> pool4 -> 6x6
        self.fc1 = nn.Linear(in_features=16 * 6 * 6, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.pool2(torch.relu(self.conv1(x)))
        x = self.pool4(torch.relu(self.conv3(x)))
        x = x.view(x.shape[0], -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
### MyCNN 

class mycnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2) # 32x32
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=5, stride=2, padding=2) # 16x16
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, padding=2)
        self.pool3 = nn.MaxPool2d(kernel_size=5, stride=2, padding=2) # 8x8
        
        self.skip1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1)
        self.skip2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=1)
    
        self.fc1 = nn.Linear(in_features=192 * 8 * 8, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=10)
        self.dropout_conv = nn.Dropout2d(0.2)
        self.dropout_fc = nn.Dropout(0.2)
    
    def forward(self, x):
        out1 = self.pool1(self.dropout_conv(torch.relu(self.conv1(x))))

        out2 = torch.relu(self.conv2(out1))
        out2 = self.dropout_conv(out2)
        out2 = self.pool2(out2 + self.skip1(out1))

        out3 = torch.relu(self.conv3(out2))
        out3 = self.dropout_conv(out3)
        out3 = self.pool3(out3 + self.skip2(out2))

        x = out3.view(out3.shape[0], -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout_fc(x)  
        x = torch.relu(self.fc2(x))
        x = self.dropout_fc(x)
        x = self.fc3(x)
        return x