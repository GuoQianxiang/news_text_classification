import torch
import torch.nn as nn
import model.MLP as MLP
from model.MLP import NeuralNet

if __name__ == '__main__':
    # step 1: prepare dataset and create dataloader
    train_loader, test_loader = MLP.create_dataloader()

    # step 2: instantiate neural network and design model
    model = NeuralNet()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # step 3: train the model
    MLP.train(train_loader, model, criterion, optimizer, num_epochs=25)

    # step 4: test the model
    MLP.test(test_loader, model)
