import torch
from src.cnn.models import mycnn, LeNet, CNN1, CNN2, CNN3, CNN4
from src.utils.metrics import evaluate_model
from torchvision import datasets, transforms


##HYPERPARAMETERS
lr = 1e-3
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_cnn_model(MODEL_NAME):
    ###test dataset
    test_data = datasets.CIFAR10(root="./data", train=False, download=True, transform=transforms.ToTensor())

    model = MODEL_NAME.to(device)
    weights = torch.load(f"data/models/{MODEL_NAME}.pth", map_location=device)

    model.load_state_dict(weights)
    model.eval()

    evaluate_model(model, test_data, batch_size=batch_size)
