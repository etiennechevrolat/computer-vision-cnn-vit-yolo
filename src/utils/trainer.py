import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from cnn.models import CNN1, CNN2, CNN3, CNN4, LeNet, mycnn
from torch.utils.tensorboard import SummaryWriter
import os
import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Accuracy 
def success_rate(model, test_data):
    model.eval()
    correct = 0
    total = 0

    dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    with torch.no_grad():
        for x, y in dataloader:
            # send data to same device as model
            x = x.to(device)
            y = y.to(device)

            y_pred = torch.argmax(model(x), dim=1)
            if (y_pred == y).item():
                correct += 1
            total += 1

    return correct / total if total > 0 else 0.0

### Training 

def trainer(dataset, model, optimizer, loss_fn, epochs=10, batch_size=1, rate=1e-4, run_name="default_run"):
	model.train()
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
	log_dir = os.path.join(os.path.dirname(__file__), "../../logs", run_name)
	writer = SummaryWriter(log_dir=log_dir)
	
	for epoch in range(epochs):
		progress_bar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
		for x, y in progress_bar:
			optimizer.zero_grad()
			# envoyer les données sur le bon device
			x = x.to(device)
			y = y.to(device)

			y_pred = model(x)
			loss = loss_fn(y_pred, y)
			loss.backward()
			optimizer.step()
			progress_bar.set_postfix(loss=loss.item())
			writer.add_scalar("Loss/train", loss.item(), epoch * len(dataloader) + progress_bar.n)
	writer.close()

