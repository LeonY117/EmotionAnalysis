import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

class net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden=1):
        super(net, self).__init__()
        self.num_hidden = num_hidden
        self.fc1 = nn.Linear(input_size, hidden_size)
        if num_hidden > 0:
            for i in range(num_hidden):
                # add non-linearity
                self.add_module('relu'+str(i+1), nn.ReLU())
                self.add_module('fc'+str(i+2), nn.Linear(hidden_size, hidden_size))
        self.output = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        if self.num_hidden > 0:
            for i in range(self.num_hidden):
                x = self._modules['relu'+str(i+1)](x)
                x = self._modules['fc'+str(i+2)](x)
        x = self.output(x)
        x = self.softmax(x)
        return x
    
class MyDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __getitem__(self, index):
        row = self.dataframe.iloc[index].to_numpy()
        row = torch.from_numpy(row).float()
        features = row[:7]
        labels = row[7:]
        return features, labels

    def __len__(self):
        return len(self.dataframe)
    
def train(model, train_loader, valid_loader, epochs=10):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            y_pred = model(x)
            train_loss = criterion(y_pred, y)
            train_loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(valid_loader):
                y_pred = model(x)
                valid_loss = criterion(y_pred, y)
        if (epoch+1) % 10 == 0:
            print('Epoch: {}, Train Loss: {:.4f}, Valid Loss: {:.4f}'\
                .format(epoch+1, train_loss, valid_loss))
    return model