import torch
from src.cnn.models import mycnn, LeNet, CNN1, CNN2, CNN3, CNN4
from src.utils.metrics import evaluate_model
from torchvision import datasets, transforms


##HYPERPARAMETERS
lr = 1e-3
MODEL_NAME = "mycnn"

###test dataset
test_data = datasets.CIFAR10(root="./data", train=False, download=True, transform=transforms.ToTensor())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MODEL_NAME().to(device)
weights = torch.load("data/models/mycnn.pth", map_location=device)

model.load_state_dict(weights)
model.eval()

evaluate_model(model, test_data, batch_size=batch_size)
