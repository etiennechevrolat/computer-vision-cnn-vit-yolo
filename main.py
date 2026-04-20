import sys
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.cnn.models import CNN1, CNN2, CNN3, CNN4, LeNet, mycnn
from src.vit.utils_vit.vision_transformer import ViT
from src.vit.utils_vit.trainer_vit import vit_trainer, evaluate_vit 
from src.cnn.utils.trainer_cnn import trainer
from src.cnn.utils.metrics import evaluate_model

#HYPERPARAMETERS
training = True
VIT = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 100
batch_size = 64
lr = 1e-3


#Datasets
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

train_data = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transforms)
test_data = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transforms)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

#cnn models 
CNN_MODELS_CONFIG = {
    "mycnn": mycnn,
    "lenet": LeNet,
    "cnn1": CNN1,
    "cnn2": CNN2,
    "cnn3": CNN3,
    "cnn4": CNN4,
}
MODEL_NAME = "mycnn"

model = ViT(img_size=32, patch_size=4, hidden_dim=128, n_heads=8, n_layers=12, dropout_rate=0.1).to(device) if VIT else CNN_MODELS_CONFIG[MODEL_NAME]().to(device)

# training
if training:
    if VIT : 
        vit_trainer(train_data, test_data, model, epochs=epochs, batch_size=batch_size, lr=3e-4, weight_decay=1e-2, run_name="ViT_CIFAR10")
    else:   
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = torch.nn.CrossEntropyLoss()
        trainer(train_data, model, optimizer, loss_fn, epochs=epochs, batch_size=batch_size, rate=lr, run_name=MODEL_NAME)

#eval 
if VIT:
    evaluate_vit(model, test_loader, device)
else:
    evaluate_model(model, test_data, batch_size=batch_size)

#saving
if VIT:
    save_dir = Path("data/models")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"vit1.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Modèle ViT1 sauvegardé sous {save_path}")
else: 
    save_dir = Path("data/models")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{MODEL_NAME}.pth"

    torch.save(model.state_dict(), save_path)
    print(f"Modèle {MODEL_NAME} sauvegardé sous {save_path}")