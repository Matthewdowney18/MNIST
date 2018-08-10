import torch
import torch.nn as nn
import torch.nn.functional as F
from mnist import MNIST
from matplotlib import pyplot


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        #self.conv1 = nn.Conv2d(1, 6, 5)
        #self.conv1 = nn.Conv2d(6, 16, 5)
        self.layer1 = nn.Linear(784, 100)
        self.layer2 = nn.Linear(100, 100)
        self.layer3 = nn.Linear(100, 10)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.05)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer2(x))
        x = torch.sigmoid(self.layer3(x))
        return x

    def backward(self, y_pred, y_train):
        loss = self.criterion(y_pred, y_train)

        # Zero the gradients
        self.optimizer.zero_grad()

        # perform a backward pass (backpropagation)
        loss.backward()

        # Update the parameters
        self.optimizer.step()
        return loss


def make_one_hot(data):
    onehot = torch.FloatTensor(len(data), 10).zero_()
    for example, num in enumerate(data):
        onehot.index_put_((torch.tensor(example, dtype=torch.long),
                             torch.tensor(int(num), dtype=torch.long)),
                            torch.tensor(1, dtype=torch.float))
    return onehot


def get_results(test, actual):
    num_correct = 0
    _, predicted = torch.max(test, 1)
    for dim in range(0, len(predicted)):
        if int(predicted[dim]) == int(actual[dim]):
            num_correct+=1
        print(int(predicted[dim]), int(actual[dim]), num_correct)
    return num_correct/len(predicted)


def plot(total_loss):
    pyplot.plot(total_loss)
    pyplot.show()


def main():
    mn_data = MNIST(
        '/home/mattd/PycharmProjects/MNIST_classifier/MNIST_dataset/')

    x_train, y_train = mn_data.load_training()
    x_test, y_test = mn_data.load_testing()
    x_train = torch.tensor(x_train, dtype=torch.float, requires_grad=True)
    y_train = torch.tensor(y_train, dtype=torch.float, requires_grad=True)
    x_test = torch.tensor(x_test, dtype=torch.float, requires_grad=True)
    y_test = torch.tensor(y_test, dtype=torch.float, requires_grad=True)

    classifier = Model()

    y_train_one = make_one_hot(y_train)

    total_loss = []
    for epoch in range(20):
        # Forward Propagation
        y_pred = classifier.forward(x_train)

        loss = classifier.backward(y_pred, y_train_one)

        print('epoch: ', epoch, ' loss: ', loss.item())
        total_loss.append(loss)

    plot(total_loss)

    test_pred = classifier.forward(x_test)

    print('accuracy: ' + str(get_results(test_pred, y_test)))

main()
