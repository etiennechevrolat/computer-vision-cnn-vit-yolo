import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from torchvision import datasets, transforms, models 
from cnn.models import CNN1, CNN2, CNN3, CNN4, LeNet, mycnn
from utils.trainer import trainer
import torch  
from utils.metrics import evaluate_model

training = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##HYPERPARAMETERS
epochs = 50
batch_size = 64
lr = 5e-3
MODEL_NAME = "mycnn"

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

##DATASETS
train_data = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transforms)
test_data = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transforms)  

##Config 
model = mycnn().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
loss_fn = torch.nn.CrossEntropyLoss()

### Training: 
if training:
    trainer(train_data, model, optimizer, loss_fn, epochs=epochs, batch_size=batch_size, rate=lr, run_name=MODEL_NAME)

### Testing:
evaluate_model(model, test_data, batch_size=batch_size)

###Save the model
torch.save(model.state_dict(), "data/models/mycnn.pth")
print("Modèle sauvegardé sous mycnn.pth")
