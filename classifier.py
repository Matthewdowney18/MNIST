import torch
import torch.nn as nn
import torch.nn.functional as F
from mnist import MNIST
from matplotlib import pyplot
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dset

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.005)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def backward(self, y_pred, target):
        loss = self.criterion(y_pred, target)

        # Zero the gradients
        self.optimizer.zero_grad()

        # perform a backward pass (backpropagation)
        loss.backward()

        # Update the parameters
        self.optimizer.step()
        return loss

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


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
    path = '/home/mattd/PycharmProjects/MNIST_classifier/data'

    trans = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_set = dset.MNIST(root=path, train=True, transform=trans,
                           download=True)
    test_set = dset.MNIST(root=path, train=False, transform=trans,
                          download=True)

    batch_size = 100

    x_train = train_set.train_data.float()
    y_train = train_set.train_labels
    x_test = test_set.test_data.float()
    y_test = test_set.test_labels
    x_train = x_train.view(60000, 1, 28, 28)
    x_test = x_test.view(10000, 1, 28, 28)


    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False)

    classifier = Model()
    print(classifier.parameters())
    y_train_one = make_one_hot(y_train)

    total_loss = []
    for epoch in range(10):
        # Forward Propagation

        y_pred = classifier.forward(x_train)

        loss = classifier.backward(y_pred, y_train_one)

        print('epoch: ', epoch, ' loss: ', loss.item())
        total_loss.append(loss)

    plot(total_loss)

    test_pred = classifier.forward(x_test)

    print('accuracy: ' + str(get_results(test_pred, y_test)))

main()
