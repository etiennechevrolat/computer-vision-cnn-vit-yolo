import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import tqdm

def vit_trainer(train_data, test_data, model, epochs=20, batch_size=64, lr=3e-4, weight_decay=1e-2, run_name="ViT_CIFAR10"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Préparation du modèle : Ajouter une tête de classification
    # Ton ViT renvoie le CLS token [batch, hidden_dim]. On doit projeter sur 10 classes.
    head = nn.Linear(model.hidden_dim, 10).to(device)
    model = model.to(device)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

    optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    log_dir = os.path.join(os.path.dirname(__file__), "../../logs/vit", run_name)
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(epochs):
        model.train()
        head.train()
        total_loss = 0
        
        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)

            # Forward
            # On passe l'image dans le ViT -> récupère le CLS token -> passe dans le head
            features = model(x)
            logits = head(features)
            loss = loss_fn(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            # Logs périodiques
            if batch_idx % 50 == 0:
                step = epoch * len(train_loader) + batch_idx
                writer.add_scalar("Loss/train", loss.item(), step)
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        # Sauvegarde de sécurité
        torch.save({
            'model_state_dict': model.state_dict(),
            'head_state_dict': head.state_dict(),
        }, f"checkpoints/{run_name}_latest.pth")

    writer.close()

def evaluate_vit(model, head, loader, device):
    model.eval()
    head.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = head(model(x))
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return 100 * correct / total