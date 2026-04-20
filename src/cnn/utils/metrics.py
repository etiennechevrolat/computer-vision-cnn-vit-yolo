import torch
import time
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model(model, test_dataset, batch_size=32):
    """
    Évalue les métriques : Accuracy, Temps d'inférence et nb de paramètres.
    """
    model.to(device)
    model.eval()
    
    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    correct = 0
    total = 0
    start_time = time.time()

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == y).sum().item()
            total += y.size(0)
    
    end_time = time.time()
    
    accuracy = 100 * correct / total
    total_time = end_time - start_time
    time_per_sample = (total_time / total) * 1000 # en ms
    
    num_params = sum(p.numel() for p in model.parameters())
    model_size_mb = (num_params * 4) / (1024**2) # float32 = 4 bytes

    print("-" * 30)
    print(f"Résultats pour le modèle : {model.__class__.__name__}")
    print("-" * 30)
    print(f"Accuracy      : {accuracy:.2f}%")
    print(f"Inférence/img : {time_per_sample:.4f} ms")
    print(f"Taille        : {model_size_mb:.2f} MB ({num_params:,} paramètres)")
    print("-" * 30)

    return {
        "accuracy": accuracy,
        "ms_per_img": time_per_sample,
        "params": num_params
    }