import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

class labelNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden=1):
        super().__init__()
        self.num_hidden = num_hidden
        self.fc1 = nn.Linear(input_size, hidden_size)
        if num_hidden > 0:
            for i in range(num_hidden):
                # add non-linearity
                self.add_module('tanh'+str(i+1), nn.Tanh())
                self.add_module('fc'+str(i+2), nn.Linear(hidden_size, hidden_size))
        self.tanh = nn.Tanh()
        self.output = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(-1)
        
    def forward(self, x):
        x = self.fc1(x)
        if self.num_hidden > 0:
            for i in range(self.num_hidden):
                x = self._modules['tanh'+str(i+1)](x)
                x = self._modules['fc'+str(i+2)](x)
        x = self.tanh(x)
        x = self.output(x)
        x = self.softmax(x)
        return x
    
class MyDataset(Dataset):
    def __init__(self, dataframe, features, labels):
        self.dataframe = dataframe
        self.features = features
        self.labels = labels

    def __getitem__(self, index):
        row = self.dataframe.iloc[index].to_numpy()
        row = torch.from_numpy(row).float()
        features = row[:len(self.features)]
        labels = torch.softmax(row[len(self.features):], -1)
        #labels = torch.nn.functional.one_hot(labels, num_classes=6).float()
        return features, labels

    def __len__(self):
        return len(self.dataframe)
    
def class_accuracy(y_pred, y_true):
    y_pred = torch.argmax(y_pred, dim=1)
    y_true = torch.argmax(y_true, dim=1)
    correct = torch.sum(y_pred == y_true)
    return correct, len(y_true)
    
def train(model, train_loader, valid_loader, lr=0.001, epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    output_freq = epochs//10
    training_loss = []
    valid_loss = []
    for epoch in range(epochs):
        #model.train()
        running_loss = 0.0
        total = 0.0
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            y_pred = model(x)
            train_loss = criterion(y_pred, y)
            train_loss.backward()
            running_loss += train_loss.item()
            total += len(y)
            optimizer.step()
        training_loss.append(running_loss/total)
        with torch.no_grad():
            correct, total = 0, 0
            val_loss = 0.0
            for i, (x, y) in enumerate(valid_loader):
                y_pred = model(x)
                val_loss += criterion(y_pred, y).item()
                cor, tot = class_accuracy(y_pred, y)
                correct += cor
                total += tot
            class_acc = correct/total
            valid_loss.append(val_loss/total)
        #model.eval()
        if (epoch+1) % output_freq == 0:
            print('Epoch: {}, Train Loss: {:.4f}, Valid loss: {:.4f}, Valid class acc: {:.4f}'\
                .format(epoch+1, training_loss[-1], valid_loss[-1], class_acc))
    return model, training_loss, valid_loss