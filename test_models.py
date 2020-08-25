import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from api import Benchmark
from utils.data_engineering import read_data, remove_uninformative_features
from utils.preprocessing import DictToTensor, extractDataAsTensor
from utils.plots import plot_evolution
from models.dnn import Network_lstm
from models.model import Model

#Loading the data set
test_temporal_data = torch.load('data/test_temporal_data.pt')
test_data_tensor = torch.load('data/test_data_tensor.pt')
test_targets = torch.load('data/test_targets.pt')

path = "./results/models/lstm_hidden3_"
path_models = "./results/models/lstm_hidden3_numlayer4_4900.pkl"

input_config_size = test_data_tensor.shape
input_temporal_size = test_temporal_data.shape
hidden_lstm_dim = 3
num_layer = 4

net = Network_lstm(input_config_size, input_temporal_size, hidden_lstm_dim, num_layer)

max_epochs = 5000
batch_size = 64

loss = nn.MSELoss(reduction='mean')
optimizer = optim.Adam(net.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
step_lr = max_epochs/10
gamma = 0.6


model = Model(net, max_epochs, batch_size, loss, optimizer, scheduler, step_lr, gamma, path, path_model=path_models)
model.test(test_data_tensor, test_temporal_data, test_targets)