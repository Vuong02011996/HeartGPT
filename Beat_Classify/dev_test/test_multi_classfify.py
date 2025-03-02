import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from sklearn.datasets import load_iris
import numpy as np
from sklearn.model_selection import train_test_split

torch.manual_seed(2)

iris = load_iris()
X = iris.data
Y = iris.target

x, x_val, y, y_val = train_test_split(X, Y, test_size=0.33, random_state=42)
print(x.shape, y.shape, x_val.shape, y_val.shape)

x_train = x.reshape(-1, x.shape[1]).astype('float32')
y_train = y

x_val = x_val.reshape(-1, x_val.shape[1]).astype('float32')
y_val = y_val

x_val = torch.from_numpy(x_val)
y_val = torch.from_numpy(y_val)



class Data(Dataset):
    def __init__(self):
        self.x=torch.from_numpy(x_train)
        self.y=torch.from_numpy(y_train)
        self.len=self.x.shape[0]
    def __getitem__(self,index):
        return self.x[index], self.y[index]
    def __len__(self):
        return self.len

data_set=Data()

trainloader=DataLoader(dataset=data_set,batch_size=64)

print("data_set.x[1:10]: ", data_set.x[1:10])
print("data_set.y[1:10]: ", data_set.y[1:10])
print("data_set.x.shape, data_set.y.shape", data_set.x.shape, data_set.y.shape)


class Net(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)

    def forward(self, x):
        x = self.linear1(x) # x (B, T) = 64, 4 => 64, 25
        x = torch.sigmoid(x) # 64, 25 -> 64, 25
        x = self.linear2(x) # 64, 25 -> 64, 3
        return x

input_dim=4     # how many Variables are in the dataset
hidden_dim = 25 # hidden layers
output_dim=3    # number of classes


# Instantiate model
model=Net(input_dim,hidden_dim,output_dim)

print('W:',list(model.parameters())[0].size())
print('b',list(model.parameters())[1].size())

criterion=nn.CrossEntropyLoss()

learning_rate=0.1
optimizer=torch.optim.SGD(model.parameters(), lr=learning_rate)

n_epochs = 1000
loss_list = []

# n_epochs
for epoch in range(n_epochs):
    for x, y in trainloader:
        # clear gradient
        optimizer.zero_grad()
        # make a prediction
        z = model(x)
        # calculate loss, da Cross Entropy benutzt wird muss ich in den loss Klassen vorhersagen,
        # also Wahrscheinlichkeit pro Klasse. Das mach torch.max(y,1)[1])
        loss = criterion(z, y) # x = z = (64, 3) ; y = (64, )
        print("loss: ", loss)
        # calculate gradients of parameters
        loss.backward()
        # update parameters
        optimizer.step()

        loss_list.append(loss.data)

        # print('epoch {}, loss {}'.format(epoch, loss.item()))