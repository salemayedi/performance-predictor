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
from models.dnn import regressor_classifier
from models.model_kfold_cross_validation import Model

print("hello")
#Loading the data set
config_test= torch.load('./data/config_test.pt')#.unsqueeze(2)
meta_test = torch.FloatTensor(torch.load('./data/meta_test.pt'))#.unsqueeze(2)
y_test = torch.load( './data/y_test.pt')



path = "./results/models/class_reg_"
path_models = "./results/models/class_reg_900.pkl"

#Initialize CNN
config_size = config_test.shape[1]
meta_size = meta_test.shape[1]

net = regressor_classifier(config_size, meta_size)


#Name the model to train
name_train = "class_reg_"
path = "./results/models/"+name_train

#Set Hyperparameters for the model
max_epochs = 1000
batch_size = 64
kfolds = 1
loss_reg = nn.MSELoss(reduction='mean')
loss_class = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
step_lr = max_epochs/10
gamma = 0.6
hidden_lstm = 3

model = Model(net, max_epochs, batch_size, loss_reg, loss_class, optimizer, scheduler, step_lr, gamma, kfolds, path, path_models)
model.test(config_test, meta_test, y_test)


