import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tqdm import tqdm
import copy

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
#scaler = MinMaxScaler(feature_range=(0,1))
scaler = StandardScaler()

"""#Data"""

dataset = pd.read_csv('https://raw.githubusercontent.com/ShawnPatrick-Barhorst/OSRS_GE_LSTM/main/OSRS_GE_price_dataset.csv')
dataset

train = dataset['Anchovy pizza'][:5500].ewm(com=10).mean()
train = train.pct_change()
train = train.fillna(method = 'bfill')
test = dataset['Pineapple pizza'][:5500].ewm(com=10).mean()
test = test.pct_change()
test.dropna()

train = np.array(train)
train = np.reshape(train, (-1, 1))
scaled_data = scaler.fit_transform(train)

window_size = 30
x_train, y_train = [], []
for i in range(window_size,5500):
    x_train.append(scaled_data[i-window_size:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

"""
#Pytorch LSTM
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torch.functional as F
import torch.optim

from torch.optim import Adam

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class data(Dataset):
  def __init__(self, exclusion_set, split_ratio, train, window_length = 30):
    dataset = pd.read_csv('https://raw.githubusercontent.com/ShawnPatrick-Barhorst/OSRS_GE_LSTM/main/OSRS_GE_price_dataset.csv')
    dataset = dataset.astype('float64')
    dataset = dataset.drop(columns=exclusion_set)
    
    #dataset = dataset.iloc[:, :1000]

    dataset = dataset.replace([np.inf, -np.inf], np.nan)
    dataset = dataset.ffill()
    dataset = dataset.bfill()

    #Shuffle and Split dataframe
    shuffled_columns = np.random.permutation(dataset.columns)
    split_index = int(len(dataset.columns) * split_ratio)

    dataset = dataset[shuffled_columns]

    if train == True:
      shuffled_columns = shuffled_columns[:split_index]
    else:
      shuffled_columns = shuffled_columns[split_index:]

    dataset = dataset[shuffled_columns]

    x_train, y_train = [], []
    for item in dataset:
      train = dataset[item][:5500].ewm(com=10).mean()
      train = train.pct_change()
      train = train.replace([np.inf, -np.inf], np.nan)
      train = train.bfill()
      train = train.ffill()
      train = np.array(train)
      train = np.reshape(train, (-1, 1))


      scaled_data = scaler.fit_transform(train)

      noise = np.random.normal(0, 0.01, size=scaled_data.shape)
      scaled_data = scaled_data + noise


      for i in range(window_length,len(scaled_data)):
        x_train.append(scaled_data[i-window_length:i,0])
        y_train.append(scaled_data[i,0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    x_train = x_train.reshape(-1, window_length, 1)
    y_train = y_train.reshape(-1, 1)

    self.x = torch.from_numpy(x_train).float()
    self.y = torch.from_numpy(y_train).float()



    print(x_train.shape)
    print(y_train.shape)

  def __len__(self):
    return len(self.x)

  def __getitem__(self, idx):
    return self.x[idx], self.y[idx]

  def get_validation_list(self):
    return self.shuffled_columns

exclusion_set = ['Anchovy pizza', 'Pineapple pizza', 'Raw tuna', 'Raw anchovies']
train_data = data(exclusion_set, train=True, split_ratio=0.8)
validation_data = data(exclusion_set, train=False, split_ratio=0.8)

batch_size = 16

trainloader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)
validloader = torch.utils.data.DataLoader(validation_data, batch_size, shuffle=True)

"""data is in format (window sample, time steps, features)"""

class LSTM_block(nn.Module):

  def __init__(self, input_dim, hidden_dim):
    super(LSTM_block, self).__init__()

    self.lstm_layer = nn.LSTM(input_dim, hidden_dim, batch_first=True)
    self.batch_norm = nn.BatchNorm1d(hidden_dim)
    self.leaky_relu = nn.LeakyReLU()
    self.dropout = nn.Dropout(0.3)

  def forward(self, x):
    x, _ = self.lstm_layer(x)
    x = self.batch_norm(x.permute(0, 2, 1))
    x = x.permute(0, 2, 1)
    x = self.leaky_relu(x)
    x = self.dropout(x)

    return x


class Dense_block(nn.Module):

  def __init__(self, input_dim, hidden_dim):
    super(Dense_block, self).__init__()

    self.linear_layer = nn.Linear(input_dim, hidden_dim)
    self.batch_norm = nn.BatchNorm1d(hidden_dim)
    self.leaky_relu = nn.LeakyReLU()
    self.dropout = nn.Dropout(0.3)

  def forward(self, x):
    x = self.linear_layer(x)
    x = self.batch_norm(x.permute(0, 2, 1))
    x = x.permute(0, 2, 1)
    x = self.leaky_relu(x)
    x = self.dropout(x)

    return x


class LSTM_model(nn.Module):

  def __init__(self, input_dim, hidden_dim):
    super(LSTM_model, self).__init__()
    self.hidden_dim = hidden_dim

    self.lstm_1 = LSTM_block(input_dim, hidden_dim)
    self.lstm_2 = LSTM_block(hidden_dim, hidden_dim)
    self.lstm_3 = LSTM_block(hidden_dim, hidden_dim)

    self.dense_1 = Dense_block(hidden_dim, hidden_dim)
    self.dense_2 = Dense_block(hidden_dim, hidden_dim)

    self.linear1 = nn.Linear(hidden_dim, 1)

  def forward(self, x):
    x = self.lstm_1(x)
    #x = self.lstm_2(x)
    #x = self.lstm_3(x)

    x = self.dense_1(x)
    x = self.dense_2(x)

    x = x[:, -1, :]

    x = self.linear1(x)

    return x

model = LSTM_model(1, 32)

class LogCosH(nn.Module):
    def __init__(self):
        super(LogCosH, self).__init__()

    def forward(self, y_true, y_pred):
        loss = torch.log(torch.cosh(y_pred - y_true))
        return torch.mean(loss)

epochs = 30
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
criterion = LogCosH()

model.to(device)

def train_model(model, trainloader, validloader, epochs, device, threshold=0.01, stop_patience=5):
  best_val_loss = float('inf')


  for epoch in range(epochs+1):
    model.train()
    total_loss = 0

    for train_x, train_y in tqdm(trainloader, desc=f'Epoch {epoch+1}/{epochs}', unit='batch', ncols=100):
      train_x, train_y = train_x.to(device), train_y.to(device)

      optimizer.zero_grad()
      outputs = model(train_x)
      loss = criterion(outputs, train_y)
      loss.backward()
      optimizer.step()

      total_loss += loss.item() * train_x.size(0)

    avg_loss = total_loss/len(trainloader)
    print('Training Loss: {:.6f}'.format(
        avg_loss
        ))

    #validation loss
    total_loss = 0
    with torch.no_grad():
      for val_x, val_y in validloader:
        val_x, val_y = val_x.to(device), val_y.to(device)

        outputs = model(val_x)
        val_loss = criterion(outputs, val_y)
        total_loss += val_loss.item() * val_x.size(0)

      avg_loss = total_loss/len(validloader)
      print('Validation Loss: {:.6f}'.format(
            avg_loss
            ))

      #Early Stopping
      if avg_loss < best_val_loss + threshold:
        best_val_loss = avg_loss
        patience = 0
        best_model_weights = copy.deepcopy(model.state_dict())
      else:
        patience += 1
        model.load_state_dict(best_model_weights)
        if patience == stop_patience:
          break

train_model(model, trainloader, validloader, epochs, device)

torch.save(model.state_dict(), 'LSTM_model.pt')

new_model = LSTM_model(1, 32)
new_model.load_state_dict(torch.load('LSTM_model.pt'))
new_model.to(device)




