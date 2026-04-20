import torch
import torch.nn as nn

### CNN 1
class CNN1(nn.Module):  
	def __init__(self): 
		super().__init__() 
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, padding=2)
		self.pool1 = nn.MaxPool2d(kernel_size=5, stride=2, padding=2)
		self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=5, padding=2)
		self.pool2 = nn.MaxPool2d(kernel_size=5, stride=2, padding=2)
		self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, padding=2, stride=1)
		self.fc1 = nn.Linear(in_features=16 * 7 * 7, out_features=10) 

	def forward(self, x): 
		x = self.conv1(x)
		x = torch.relu(x)
		x = self.pool1(x)


		x = self.conv2(x)
		x = self.pool2(x)
		x = torch.relu(x)
            
### Influence of the depth of the filters

# CNN 2
class CNN2(nn.Module):  # a class inheriting from nn.Module
	def __init__(self): 
		super().__init__()  # call the constructor of nn.Module
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=5, padding=2)
		self.pool1 = nn.MaxPool2d(kernel_size=5, stride=2, padding=2)
		self.conv2 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=5, padding=2)
		self.pool2 = nn.MaxPool2d(kernel_size=5, stride=2, padding=2)
		self.conv3 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=5, padding=2, stride=1)
		self.fc1 = nn.Linear(in_features=8 * 7 * 7, out_features=10) 

	def forward(self, x): 
		# how do the layers compute the output ?
		# this function needs to return the output of the net
		# usually by applying the layers in the right order
		x = self.conv1(x)
		x = torch.relu(x)
		x = self.pool1(x)


		x = self.conv2(x)
		x = self.pool2(x)
		x = torch.relu(x)


		x = self.conv3(x)  

		x = x.view( x.shape[0], -1)  # flatten the tensor
		x = self.fc1(x)

		return x


		x = self.conv3(x)  

		x = x.view( x.shape[0], -1)  # flatten the tensor
		x = self.fc1(x)

		
		return x

### Filters of smaller size 

# CNN 3
class CNN3(nn.Module):  # a class inheriting from nn.Module
	def __init__(self): 
		super().__init__()  # call the constructor of nn.Module
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding=1) ##(1,28,28) -> (2, 28, 28)
		self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) ## (2, 28, 28) -> (2, 14, 14)
		self.conv2 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, padding=1) ## (2, 14, 14) -> (4, 14, 14)
		self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) ## (4, 14, 14) -> (4, 7, 7)
		self.conv3 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1, stride=1) ## (4, 7, 7) -> (8, 7, 7)
		self.fc1 = nn.Linear(in_features=8 * 7 * 7, out_features=10) 

	def forward(self, x): 
		# how do the layers compute the output ?
		# this function needs to return the output of the net
		# usually by applying the layers in the right order
		x = self.conv1(x)
		x = torch.relu(x)
		x = self.pool1(x)


		x = self.conv2(x)
		x = self.pool2(x)
		x = torch.relu(x)


		x = self.conv3(x)  

		x = x.view( x.shape[0], -1) 
		x = self.fc1(x)

		return x


### CNN 4 - One conv layer 

class CNN4(nn.Module):
    def __init__(self):
        super().__init__() 
        self.conv = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, padding=2)  ## (1,28,28) -> (4, 28, 28)
        self.fc = nn.Linear(in_features=4 * 28 * 28, out_features=10) 
    def forward(self, x): 
        x = self.conv(x)
        x = torch.relu(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
        
        ### LeNet 

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.pool2= nn.MaxPool2d(kernel_size=5, stride = 2, padding=2)
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0)
        self.pool4 = nn.MaxPool2d(kernel_size=5, stride = 2, padding=2)
        self.fc1 = nn.Linear(in_features= 16 * 5 *5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features= 84)
        self.fc3 = nn.Linear(in_features=84, out_features= 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = torch.relu(x)
        x = self.pool4(x)

        x = x.view(x.shape[0], -1)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        x = self.fc3(x)
        return x

### MyCNN 

class cnnTD3(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.pool1= nn.MaxPool2d(kernel_size=5, stride = 1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=5, stride = 2, padding=2)
        self.conv3 = nn.Conv2d(in_channels= 64, out_channels= 192, kernel_size=5, padding=2)
        self.pool3 = nn.MaxPool2d(kernel_size= 5, stride = 2, padding=2)
        self.fc1 = nn.Linear(in_features= 192 * 7 * 7, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features= 32)
        self.fc3 = nn.Linear(in_features=32, out_features= 10)
        self.dropout_conv = nn.Dropout2d(0.2)
        self.dropout_fc = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.dropout_conv(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = torch.relu(x)
        x = self.dropout_conv(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = torch.relu(x)
        x = self.dropout_conv(x)
        x = self.pool3(x)

        x = x.view(x.shape[0], -1)

        x = torch.relu(self.fc1(x))
        x = self.dropout_fc(x)  
        x = torch.relu(self.fc2(x))
        x = self.dropout_fc(x)
        x = self.fc3(x)

        return x

