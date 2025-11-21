import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


trainloader, testloader = [], []
classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def data_augmentation_transform():
    global trainloader
    global testloader

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    test_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                            shuffle=False, num_workers=2)

def transform(batchsize=4):
    global trainloader
    global testloader

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize,
                                            shuffle=False, num_workers=2)
    
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    
class OneNet(nn.Module):
    def __init__(self):
        super(OneNet, self).__init__()

        self.fc = nn.Linear(3 * 32 * 32, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
class MultipleNet(nn.Module):
    def __init__(self, relu):
        super(MultipleNet, self).__init__()

        self.relu = relu
        self.fc1 = nn.Linear(3 * 32 * 32, 110)
        self.fc2 = nn.Linear(110, 74)
        self.fc3 = nn.Linear(74, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        if (self.relu):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        else:
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.fc3(x)
        
        return x
    
class ConvultionalNet(nn.Module):
    def __init__(self):
        super(ConvultionalNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 110)
        self.fc2 = nn.Linear(110, 74)
        self.fc3 = nn.Linear(74, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    
def test(net, dataloader): 
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    images = images.to(device)
    labels = labels.to(device)
    outputs = net(images)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


def train(net, optimizer, criterion, loss_type='cross_entropy'):

    global trainloader
    global device
    running_loss = 0.0
    total = 0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        
        if loss_type == 'mse':
            labels_onehot = torch.zeros(labels.size(0), 10).to(device)
            labels_onehot.scatter_(1, labels.view(-1, 1), 1)
            loss = criterion(outputs, labels_onehot)
        else:
            loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        total += 1
        # if i % 2000 == 1999:    # print every 2000 mini-batches
        #     print('[%d, %5d] loss: %.3f' %
        #         (epoch + 1, i + 1, running_loss / 2000))
        #     running_loss = 0.0

    return running_loss / total


def train_test(epoch, net, optimizer, criterion, loss_type='cross_entropy'):
    loss_arr, train_arr, test_arr = [], [], []
    global trainloader
    global testloader
    for i in range(epoch):
        running_loss = train(net, optimizer, criterion, loss_type)

        train_acc = test(net, trainloader)
        print('[Epoch %d/50 ]: Accuracy of the network on the 10000 train images: %d %%' % (i+1, 100 * train_acc))
        
        test_acc = test(net, testloader)
        print('[Epoch %d/50 ]: Accuracy of the network on the 10000 test images: %d %%' % (i+1, 100 * test_acc))

        loss_arr.append(running_loss)
        train_arr.append(train_acc)
        test_arr.append(test_acc)
    
    return loss_arr, train_arr, test_arr
         

def plotLoss(figure, x, y):
        plt.figure(figure)
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.plot(x, y)
    

def plotAccuracy(figure, x1, y1, x2, y2):
        plt.figure(figure)
        plt.title("Training/Test Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.plot(x1, y1, label="Training")
        plt.plot(x2, y2, label="Testing")
        plt.legend()

def plotLossLearningRates(figure, x, y1, y2, y3, y4):
        plt.figure(figure)
        plt.title("Loss Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Loss"),
        plt.plot(x, y1, label="lrr=10")
        plt.plot(x, y2, label="lrr=0.1")
        plt.plot(x, y3, label="lrr=0.01")
        plt.plot(x, y4, label="lrr=0.001")
        plt.legend()

def plotLossBatchSizes(figure,x, y1, y2, y3):
        plt.figure(figure)
        plt.title("Loss Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.plot(x, y1, label="batch=1")
        plt.plot(x, y2, label="batch=4")
        plt.plot(x, y3, label="batch=1000")
        plt.legend()

criterion = nn.CrossEntropyLoss().to(device)
epoch = 35
epoch_arr = [i for i in range(1, epoch+1)]

#One neural network
transform()
oneNet = OneNet().to(device)
optimizer = optim.SGD(oneNet.parameters(), lr=0.001, momentum=0.9)
loss_arr, train_arr, test_arr = train_test(epoch, oneNet, optimizer, criterion)
plotLoss(1, epoch_arr, loss_arr)
plt.savefig('1NN Loss.png')
plotAccuracy(2, epoch_arr, train_arr, epoch_arr, test_arr)
plt.savefig('1NN Test_Train.png')

#Multiple neural network
transform()
multipleNet = MultipleNet(relu=True).to(device)
optimizer = optim.SGD(multipleNet.parameters(), lr=0.001, momentum=0.9)
loss_arr, train_arr, test_arr = train_test(epoch, multipleNet, optimizer, criterion)
plotLoss(3, epoch_arr, loss_arr)
plt.savefig('MultiNNRelu Loss.png')
plotAccuracy(4, epoch_arr, train_arr, epoch_arr, test_arr)
plt.savefig('MultiNNRelu Test_Train.png')

transform()
multipleNet = MultipleNet(relu=False).to(device)
optimizer = optim.SGD(multipleNet.parameters(), lr=0.001, momentum=0.9)
loss_arr, train_arr, test_arr = train_test(epoch, multipleNet, optimizer, criterion)
plotLoss(5, epoch_arr, loss_arr)
plt.savefig('MultiNNnoRelu Loss.png')
plotAccuracy(6, epoch_arr, train_arr, epoch_arr, test_arr)
plt.savefig('MultiNNnoRelu Test_Train.png')


# CNN
batches = [1, 4, 1000] 
learning_rates = [10, 0.1, 0.01, 0.001]


loss_batch_results, train_batch_results, test_batch_results = [], [], []
for b in batches:
    transform(b)
    cnn = ConvultionalNet().to(device)
    optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)
    loss_arr, train_arr, test_arr = train_test(epoch, cnn, optimizer, criterion)
    loss_batch_results.append(loss_arr)
    train_batch_results.append(train_arr)
    test_batch_results.append(test_arr)

plotLossBatchSizes(7, epoch_arr, loss_batch_results[0], loss_batch_results[1], loss_batch_results[2])
plt.savefig('BatchSize_Loss.png')
plotAccuracy(8, epoch_arr, train_batch_results[0], epoch_arr, test_batch_results[0])
plt.savefig('BatchSize1 Test_Train.png')
plotAccuracy(9, epoch_arr, train_batch_results[1], epoch_arr, test_batch_results[1])
plt.savefig('BatchSize4 Test_Train.png')
plotAccuracy(10, epoch_arr, train_batch_results[2], epoch_arr, test_batch_results[2])
plt.savefig('BatchSize1000 Test_Train.png')

print("-----Finished training batches-----")

loss_batch_results, train_batch_results, test_batch_results = [], [], []
for learning_rate in learning_rates:
    transform()
    cnn = ConvultionalNet().to(device)
    optimizer = optim.SGD(cnn.parameters(), lr=learning_rate, momentum=0.9)
    loss_arr, train_arr, test_arr = train_test(epoch, cnn, optimizer, criterion)
    loss_batch_results.append(loss_arr)
    train_batch_results.append(train_arr)
    test_batch_results.append(test_arr)

print(loss_batch_results)
plotLossLearningRates(11, epoch_arr, loss_batch_results[0], loss_batch_results[1], loss_batch_results[2], loss_batch_results[3])
plt.savefig('LearningRates_Loss.png')
plotAccuracy(12, epoch_arr, train_batch_results[0], epoch_arr, test_batch_results[0])
plt.savefig('LearningRate10 Test_Train.png')
plotAccuracy(13, epoch_arr, train_batch_results[1], epoch_arr, test_batch_results[1])
plt.savefig('LearningRate0.1 Test_Train.png')
plotAccuracy(14, epoch_arr, train_batch_results[2], epoch_arr, test_batch_results[2])
plt.savefig('LearningRate0.01 Test_Train.png')
plotAccuracy(15, epoch_arr, train_batch_results[3], epoch_arr, test_batch_results[3])
plt.savefig('LearningRate0.001 Test_Train.png')

print("-----Finished training learning rates -----")

data_augmentation_transform()
cnn = ConvultionalNet().to(device)
optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)
loss_arr, train_arr, test_arr = train_test(epoch, cnn, optimizer, criterion)
plotLoss(16, epoch_arr, loss_arr)
plt.savefig('DataAugmentation Loss.png')
plotAccuracy(17, epoch_arr, train_arr, epoch_arr, test_arr)
plt.savefig('DataAugmentation Test_Train.png')

print("-----Finished training data augmentation -----")

transform()
cnn = ConvultionalNet().to(device)
optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)
criterion = nn.MSELoss().to(device)
loss_arr, train_arr, test_arr = train_test(epoch, cnn, optimizer, criterion, loss_type='mse')
plotLoss(18, epoch_arr, loss_arr)
plt.savefig('MSE Loss.png')
plotAccuracy(19, epoch_arr, train_arr, epoch_arr, test_arr)
plt.savefig('MSE Test_Train.png')



