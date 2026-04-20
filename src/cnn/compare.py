import torch
from torchvision import datasets, transforms, models 
from utils.trainer import trainer
from utils.metrics import evaluate_model


### Compare with resnet18 pretrained on imagenet
model_compare = models.resnet18(pretrained=True).to(device)
# Adapter le dernier layer pour CIFAR-10 (10 classes au lieu de 1000)
model_compare.fc = torch.nn.Linear(model_compare.fc.in_features, 10)
### Entrainement de cette dernire couche pour adapter le modèle à CIFAR-10
# On gèle les autres couches pour ne pas les entraîner
for param in model_compare.parameters():
    param.requires_grad = False
for param in model_compare.fc.parameters():
    param.requires_grad = True
trainer(train_data, model_compare, torch.optim.AdamW(model_compare.fc.parameters(), lr=1e-3), torch.nn.CrossEntropyLoss(), epochs=5, batch_size=batch_size, rate=1e-3, run_name="resnet18_finetune")
model_compare = model_compare.to(device)
evaluate_model(model_compare, test_data, batch_size=batch_size)