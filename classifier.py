import torch
import torch.nn as nn
import torch.nn.functional as F
import mnist as MNIST
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
        #self.criterion = nn.MSELoss()
        #self.optimizer = torch.optim.SGD(self.parameters(), lr=0.005)

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
    '''
    def backward(self, y_pred, target):
        loss = self.criterion(y_pred, target)

        # Zero the gradients
        self.optimizer.zero_grad()

        # perform a backward pass (backpropagation)
        loss.backward()

        # Update the parameters
        self.optimizer.step()
        return loss
    '''
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def make_one_hot(data):
    onehot = torch.LongTensor(len(data), 10).zero_()
    for example, num in enumerate(data):
        onehot.index_put_((torch.tensor(example, dtype=torch.long),
                             torch.tensor(int(num), dtype=torch.long)),
                            torch.tensor(1, dtype=torch.long))
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
    pyplot.title('loss')
    pyplot.xlabel('epoch')
    pyplot.ylabel('loss')
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


    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=len(test_set),
        shuffle=False)

    classifier = Model()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01,
                                momentum=0.9)

    running_loss = 0.0
    total_loss = []
    for epoch in range(3):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            input, label = data
            output = classifier(input)

            loss = criterion(output, label)


            total_loss.append(float(loss.item()))
            # Zero the gradients
            optimizer.zero_grad()
            # perform a backward pass (backpropagation)
            loss.backward()

            # Update the parameters
            optimizer.step()

            #print('epoch: ', epoch, ' loss: ', loss.item()))

            running_loss += loss.item()
            if i % 100 == 0:# print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    plot(total_loss)

    dataiter = iter(test_loader)
    images, labels = dataiter.next()
    #images, labels = test_set

    outputs = classifier(images)
    # print images


    print('accuracy: ' + str(get_results(outputs,
        labels)))

main()
