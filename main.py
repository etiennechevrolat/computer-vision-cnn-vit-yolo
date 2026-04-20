import sys
import os


from torchvision import datasets, transforms, models 
from cnn.models import CNN1, CNN2, CNN3, CNN4, LeNet, mycnn
from vit.utils_vit.vision_transformer import ViT
from vit.utils_vit.trainer_vit import vit_trainer
from cnn.utils.trainer_cnn import trainer
from cnn.utils.metrics import evaluate_model
import torch  

# --- PARAMÈTRES ---
training = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 50
batch_size = 64
lr = 1e-3

#modèles
MODELS_CONFIG = {
    "mycnn": mycnn,
    "lenet": LeNet,
    "cnn1": CNN1,
    "cnn2": CNN2,
    "cnn3": CNN3,
    "cnn4": CNN4,
    "vit": ViT
}
MODEL_NAME = "vit"
# datasets
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

# Instanciation dynamique
model = MODELS_CONFIG[MODEL_NAME]().to(device)
if MODEL_NAME != "vit":
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = torch.nn.CrossEntropyLoss()

if training:
    if MODEL_NAME == "vit":
        # On utilise ton trainer spécifique ViT
        vit_trainer(train_data, test_data, model, epochs=20, batch_size=64, lr=3e-4, weight_decay=1e-2, run_name=MODEL_NAME)
    else:
        trainer(train_data, model, optimizer, loss_fn, epochs=epochs, batch_size=batch_size, rate=lr, run_name=MODEL_NAME)
if MODEL_NAME == "vit":
    # On utilise ton trainer spécifique ViT
    evaluate_vit(model, head, test_loader, device)
else:
    evaluate_model(model, test_data, batch_size=batch_size)

save_dir = "data/models"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f"{MODEL_NAME}.pth")

torch.save(model.state_dict(), save_path)
print(f"Modèle {MODEL_NAME} sauvegardé sous {save_path}")