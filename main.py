import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from torchvision import datasets, transforms
from cnn.models import CNN1, CNN2, CNN3, CNN4, LeNet, mycnn
from utils.trainer import trainer, success_rate
import torch  

training = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##HYPERPARAMETERS
epochs = 100
batch_size = 128
lr = 1e-3
MODEL_NAME = "mycnn"

##DATASET
train_data = datasets.CIFAR10(root="./data", train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.CIFAR10(root="./data", train=False, download=True, transform=transforms.ToTensor())  

##Config 
model = mycnn().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
loss_fn = torch.nn.CrossEntropyLoss()

### Training: 
if training:

    trainer(train_data, model, optimizer, loss_fn, epochs=epochs, batch_size=batch_size, rate=lr, run_name=MODEL_NAME)
