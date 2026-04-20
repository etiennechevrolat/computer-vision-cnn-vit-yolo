import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def vit_trainer(train_data, test_data, model, epochs=20, batch_size=64, lr=3e-4, weight_decay=1e-2, run_name="ViT_CIFAR10"):

    model.to(device)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    log_dir = os.path.join(os.path.dirname(__file__), "../../../logs/vit", run_name)
    writer = SummaryWriter(log_dir=log_dir)
    
    # Sécurité : on s'assure que le dossier checkpoints existe
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)

            logits = model(x) 
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

        torch.save({
            'model_state_dict': model.state_dict(),
        }, f"checkpoints/{run_name}_latest.pth")

    writer.close()


def evaluate_vit(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x) 
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
            
    acc = 100 * correct / total
    print(f"\nPrécision ViT sur le set de test : {acc:.2f}%")
    return acc