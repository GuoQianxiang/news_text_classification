import torch
import torch.nn as nn
import utils.utils as utils
import utils.dataloader as dataloader
from torch.utils.data import TensorDataset


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        # define the activation function
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.fc1 = nn.Linear(74, 500)

        self.fc2 = nn.Linear(500, 500)

        self.fc3 = nn.Linear(500, 500)

        self.fc4 = nn.Linear(500, 6)
        self.type = 'MLP'

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)

        x = self.fc2(x)
        x = self.sigmoid(x)
        #
        # x = self.fc3(x)
        # x = self.sigmoid(x)

        out = self.fc4(x)
        return out


def create_dataloader():
    # load the  training_data
    training_data = dataloader.load('../preprocess/converted_training.csv')

    # split training data and label
    x = training_data[:, :-1]  # 得到训练集的数据
    y = training_data[:, -1]  # 得到训练集的标签

    # shuffle the training data and split to train and test part
    x_train, y_train, x_test, y_test = utils.shuffle_data(x, y)

    # convert numpy to tensor
    tensor_x_train = torch.tensor(x_train * 100000, dtype=torch.float32)  # transform to torch tensor
    tensor_y_train = torch.tensor(y_train, dtype=torch.float32)

    tensor_x_test = torch.tensor(x_test * 100000, dtype=torch.float32)
    tensor_y_test = torch.tensor(y_test, dtype=torch.float32)

    # create dataloader
    train_data_loader = torch.utils.data.DataLoader(dataset=TensorDataset(tensor_x_train, tensor_y_train),
                                                    batch_size=50,
                                                    shuffle=True)

    test_data_loader = torch.utils.data.DataLoader(dataset=TensorDataset(tensor_x_test, tensor_y_test),
                                                   batch_size=50,
                                                   shuffle=False)

    return train_data_loader, test_data_loader


def train(train_loader, model, criterion, optimizer, num_epochs):
    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for step, (images, labels) in enumerate(train_loader):
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