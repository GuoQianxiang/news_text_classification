import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader

import model.dataloader as dataloader
import utils

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(74, 500)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(500, 6)
        self.type = 'MLP'

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        out = self.fc2(x)
        return out


def create_dataloader():
    # MNIST dataset
    training_data = dataloader.load('../preprocess/converted_training_data.csv')
    # 对引入的数据按照数据和标签进行切割
    x = training_data[:, :-1]  # 得到训练集的数据
    y = training_data[:, -1]  # 得到训练集的标签
    # shuffle the training data and split to train and test part
    x_train, y_train, x_test, y_test = utils.shuffle_data(x, y)

    tensor_x_train = torch.tensor(x_train*10000, dtype=torch.float32)  # transform to torch tensor
    tensor_y_train = torch.tensor(y_train, dtype=torch.float32)

    tensor_x_test = torch.tensor(x_test*10000, dtype=torch.float32)
    tensor_y_test = torch.tensor(y_test, dtype=torch.float32)


    # test_dataset = torchvision.datasets.MNIST(root='data',
    #                                           train=False,
    #                                           download=True,
    #                                           transform=transforms.ToTensor())

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=TensorDataset(tensor_x_train, tensor_y_train),
                                               batch_size=64,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=TensorDataset(tensor_x_test, tensor_y_test),
                                              batch_size=64,
                                              shuffle=False)

    return train_loader, test_loader


def train(train_loader, model, criterion, optimizer, num_epochs):
    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for step, (images, labels) in enumerate(train_loader):
            # print(step)
            # print(images)
            # print(labels)
            if model.type == 'MLP':
                images = images.reshape(-1, 74)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels.long())

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, step + 1, total_step, loss.item()))


def test(test_loader, model):
    # Test the model
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            if model.type == 'MLP':
                images = images.reshape(-1, 74)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))


if __name__ == '__main__':
    ### step 1: prepare dataset and create dataloader
    train_loader, test_loader = create_dataloader()

    ### step 2: instantiate neural network and design model
    model = NeuralNet()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    ### step 3: train the model
    train(train_loader, model, criterion, optimizer, num_epochs=5)

    ### step 4: test the model
    test(test_loader, model)





