import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from cnn.models import CNN1, CNN2, CNN3, CNN4, LeNet, cnnTD3
from .trainer import trainer


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


### Training of my CNN

import cloudpickle as pickle   
def torch_saver(net, file="temp"):
    with open(file, 'wb') as f:
        pickle.dump(net, f)
        
def train_my_cnn():
    model = cnnTD3().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer(training_data, model, optimizer, nn.CrossEntropyLoss(), epochs=10, batch_size=128, run_name="my_cnn_fashionmnist")
    print(f"Success rate of my CNN : {success_rate(model, test_data)}")
    torch_saver(model, file="my_cnn_fashionmnist.pkl")



### Script for training and testing the models all at once with the same hyperparameters

training = False



##HYPERPARAMETERS
epochs = 100
batch_size = 128
lr = 1e-3

def main():

    ### CNN1
    if training :
        model1 = CNN1().to(device)
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=lr)
        trainer(training_data, model1, optimizer1, nn.CrossEntropyLoss(), epochs=epochs, batch_size=batch_size, run_name="cnn1_fashionMNIST")
        print(f"Success rate of CNN1 : {success_rate(model1, test_data)}")
    
    ## Influence of the size of the filters 
    
    ### CNN2 - filters of half the depth
    if training :
        model2 = CNN2().to(device)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=lr)
        trainer(training_data, model2, optimizer2, nn.CrossEntropyLoss(), epochs=epochs, batch_size=batch_size, run_name="cnn2_fashionmnist")
        print(f"Success rate of CNN2 : {success_rate(model2, test_data)}")

    ### CNN3 - kernels of smaller size : 3x3 instead of 5x5

    if training :
        model3 = CNN3().to(device)
        optimizer3 = torch.optim.Adam(model3.parameters(), lr=lr)
        trainer(training_data, model3, optimizer3, nn.CrossEntropyLoss(), epochs=epochs, batch_size=batch_size, run_name="cnn3_fashionmnist")
        print(f"Success rate of CNN3 : {success_rate(model3, test_data)}")
    
    ## Influence of the depth of the network

    ### CNN4 - only one convolutional layer
    if training :
        model4 = CNN4().to(device)
        optimizer4 = torch.optim.Adam(model4.parameters(), lr=lr)
        trainer(training_data, model4, optimizer4, nn.CrossEntropyLoss(), epochs=epochs, batch_size=batch_size, run_name="cnn4_fashionmnist")
        print(f"Success rate of CNN4 : {success_rate(model4, test_data)}")

    ## LeNet
        if training :
            model5 = LeNet().to(device)
            optimizer5 = torch.optim.Adam(model5.parameters(), lr=lr)
            trainer(training_data, model5, optimizer5, nn.CrossEntropyLoss(), epochs=epochs, batch_size=batch_size, run_name="cnn5_fashionmnist")
            print(f"Success rate of CNN5 : {success_rate(model5, test_data)}")

if __name__ == "__main__":
    main()

