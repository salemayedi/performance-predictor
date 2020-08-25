import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from models.dnn import Network_4
from models.model import Model
from utils.plots import plot_evolution



train_temporal_data = torch.load('data/train_temporal_data.pt') #and torch.load('file.pt')
train_data_tensor = torch.load('data/train_data_tensor.pt')
train_targets = torch.load('data/train_targets.pt')
val_temporal_data = torch.load('data/val_temporal_data.pt')
val_data_tensor = torch.load('data/val_data_tensor.pt')
val_targets = torch.load('data/val_targets.pt')
test_temporal_data = torch.load('data/test_temporal_data.pt')
test_data_tensor = torch.load('data/test_data_tensor.pt')
test_targets = torch.load('data/test_targets.pt')

print(train_data_tensor.shape)
print(train_temporal_data.shape)

#Initialize CNN
input_config_size = (train_data_tensor.shape[1])
input_temporal_size = (train_temporal_data.shape[1])

net = Network_4(input_config_size, input_temporal_size)


#Name the model to train
name_train = "new_train"
path = "./results/models/"+name_train

#Set Hyperparameters for the model
max_epochs = 500
batch_size = 64
loss = nn.MSELoss(reduction='mean')
optimizer = optim.Adam(net.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
step_lr = max_epochs / 10
gamma = 0.6

model = Model(net, max_epochs, batch_size, loss, optimizer, scheduler, step_lr, gamma, path)

train_hist, val_hist = model.train(train_data_tensor, train_temporal_data, train_targets, 
                                   val_data_tensor, val_temporal_data, val_targets)

np.save('results/stats/train_hist_'+name_train, train_hist)
np.save('results/stats/val_hist_'+name_train, val_hist)
print("\n Saved history of train and validation.\n")

model.test(test_data_tensor, test_temporal_data, test_targets)

plot_evolution(train_hist, val_hist, './results/plots/'+name_train)
print("\n Saved plots.")