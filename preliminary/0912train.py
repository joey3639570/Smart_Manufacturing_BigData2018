import torch as t
from torch import nn
from torch.autograd import Variable as V
from torch import optim

#data loading
FILENAME = '/tensorvolar/joey/pywavelet_movingaverage/0903traindata.csv'
df, quality, datalist = datafunc.load_data(FILENAME,40)
quality = np.expand_dims(quality,axis=1)

#Mini Shuffle Batch
def next_batch(num, data, labels):
    '''
    Return a total of 'num' random samples and labels.
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

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

net = Net()
#Optimizer Setting
optimizer = optim.Adam(params=net.parameters(),lr=0.0001)
optimizer.zero_grad()

input = V()
output = net(input)
output.backward(output)

optimizer.step()
