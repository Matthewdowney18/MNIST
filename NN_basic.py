import torch
import torch.nn as nn
import torch.nn.functional as F
from mnist import MNIST
from matplotlib import pyplot

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
    index = torch.tensor(range(0, 10), dtype=torch.long)
    for dim in range(0, len(predicted)):

        if int(predicted[dim]) == int(actual[dim]):
            num_correct+=1
        print(predicted[dim], actual[dim], num_correct)
    return num_correct/len(predicted)

mndata = MNIST('/home/mattd/PycharmProjects/MNIST_classifier/MNIST_dataset/')

x_train, y_train = mndata.load_training()
x_test, y_test = mndata.load_testing()
x_train = torch.tensor(x_train, dtype=torch.float, requires_grad=True)
y_train = torch.tensor(y_train, dtype=torch.float, requires_grad=True)
x_test = torch.tensor(x_test, dtype=torch.float, requires_grad=True)
y_test = torch.tensor(y_test, dtype=torch.float, requires_grad=True)

n_in, n_h, n_out, batch_size = 784, 100, 10, len(x_train)

model = nn.Sequential(nn.Conv2d(n_in, n_h),
                     nn.ReLU(),
                     nn.Linear(n_h, n_out))

criterion = nn.MSELoss()


optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

y_train_one = make_one_hot(y_train)

total_loss = []
for epoch in range(10):
    # Forward Propagation
    y_pred = model(x_train)

    # Compute and print loss
    loss = criterion(y_pred, y_train_one)
    print('epoch: ', epoch, ' loss: ', loss.item())
    total_loss.append(loss.item)

    # Zero the gradients
    optimizer.zero_grad()

    # perform a backward pass (backpropagation)
    loss.backward()

    # Update the parameters
    optimizer.step()

#plot(total_loss)

test_pred = model(x_test)

print('accuracy: ' + str(get_results(test_pred, y_test)))