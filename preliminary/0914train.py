import torch as t
import torch.utils.data
from torch import nn
from torch.autograd import Variable as V
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import data.function as datafunc
import numpy as np

batch_size = 8
num_epochs = 10000

#data loading
FILENAME = '/home/cch/Downloads/pytorchworkplace/pytorch/preliminary/0903traindata.csv'
df, quality, datalist = datafunc.load_data(FILENAME,40)
quality = np.expand_dims(quality,axis=1)

def train_data_loading(input,output):
    #input should be a numpy array, e.g. datalist.values, output should be a 1d array.
    X = input
    Y = output
    #Using Output as our training label
    encoder = LabelEncoder()
    encoder.fit(Y)
    Y = encoder.transform(Y)
    #Split training set and test set to do training and test together
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    X_train, Y_train =t.FloatTensor(X_train), t.FloatTensor(Y_train)
    X_test, Y_test =t.FloatTensor(X_test), t.FloatTensor(Y_test) #Since our output is around 0.3~1.1, it should be float tensor

    #If using picture label(one hot style), label should be Long tensor

    train_dataset = t.utils.data.TensorDataset(X_train,Y_train)
    test_dataset = t.utils.data.TensorDataset(X_test,Y_test)

    return train_dataset,test_dataset

#Model definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #DNN 352 -> 176 -> 88 -> 44 -> 22 -> 11 -> 1
        #Tanh as activation function
        self.classifier = nn.Sequential(
            nn.Linear(352, 176),
            nn.Tanh(),
            nn.Linear(176, 88),
            nn.Tanh(),
            nn.Linear(88, 44),
            nn.Tanh(),
            nn.Linear(44, 22),
            nn.Tanh(),
            nn.Linear(22, 11),
            nn.Tanh(),
            nn.Linear(11, 1)
            )

    def forward(self, x):
        x = self.classifier(x)
        return x

def train(net, device, train_loader, optimizer, epoch):
    net.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = V(data), V(target)
        optimizer.zero_grad() # Must clear your gradient before training
        target = target.view(batch_size,1)
        output = net(data)
        loss = F.mse_loss(output, target) #If doing classificaiton, use nll_loss or cross_entropy
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data[0]))

def test(net, device, test_loader):
    net.eval()
    test_loss = 0
    correct = 0
    with t.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = net(data)
            target = target.view(batch_size,1)
            test_loss += F.mse_loss(output, target, reduction='sum').item() # sum up batch loss

    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}\n'.format(test_loss))

def restore_net():
    # restore entire net1 to net2
    net2 = torch.load('net.pkl')

def main():
    device = t.device("cpu")
    #Data Loader (Input Pipeline)
    train_dataset,test_dataset = train_data_loading(datalist.values,quality)
    train_loader = t.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
    test_loader = t.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
    net = Net()
    #Optimizer Setting
    optimizer = optim.Adam(params=net.parameters(),lr=0.0001)

    for epoch in range(1, num_epochs + 1):
        train(net, device, train_loader, optimizer, epoch)
        test(net, device, test_loader)
    torch.save(net, 'test.pkl')

if __name__ == '__main__':
    main()
