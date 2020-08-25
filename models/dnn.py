# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 17:42:48 2020

@author: Mikel
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

kernel_size = 3
kernel_pool = (7)
class Network_1(nn.Module):
    def __init__(self, x_dim, x_t_dim, hidden_dim=32):
        super(Network_1, self).__init__()
        
        self.conv1 = nn.Conv1d(1, 6, kernel_size)
        self.conv2 = nn.Conv1d(6, 12, kernel_size)
        #self.conv3 = nn.Conv1d(12,24, kernel_size)
        self.fc_conv = nn.Linear(12, 1)
        
        
        
        self.fc1 = nn.Linear(x_dim+x_t_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1) 

        self.dropout = nn.Dropout(p=0.33)
        self.bn1 = nn.BatchNorm1d(12)
    
    def forward(self, x, x_t):
        
        x_t = self.conv1(x_t)
        x_t = self.conv2(x_t)
        x_t = F.relu(self.bn1(x_t))
        x_t = F.max_pool1d(x_t, kernel_pool)
        x_t = x_t.view(x_t.shape[0], -1)
        x_t = F.relu(self.fc_conv(x_t))
        
        
        x_cat = torch.cat((x_t, x), 1)
        
        x_cat = F.relu(self.fc1(x_cat))
        x_cat = self.dropout(x_cat)
        x_cat = F.relu(self.fc2(x_cat))
        x_cat = self.dropout(x_cat)
        output = F.relu(self.fc3(x_cat))
       
        return output

class Network_2(nn.Module):
    def __init__(self, x_dim, x_t_dim, hidden_dim=32):
        super(Network_2, self).__init__()
        
        self.conv1_a = nn.Conv1d(1, 6, kernel_size)
        self.conv2_a = nn.Conv1d(6, 12, kernel_size)
        self.fc_conv_a = nn.Linear(12, 1)
        
        self.conv1_b = nn.Conv1d(1, 6, kernel_size)
        self.conv2_b = nn.Conv1d(6, 12, kernel_size)
        self.fc_conv_b = nn.Linear(12, 1)
        
        
        
        self.fc1 = nn.Linear(x_dim+x_t_dim*2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1) 

        self.dropout = nn.Dropout(p=0.33)
        self.bn1 = nn.BatchNorm1d(12)
    
    def forward(self, x, x_t_a, x_t_b):
        
        x_t_a = self.conv1_a(x_t_a)
        x_t_a = self.conv2_a(x_t_a)
        x_t_a = F.relu(self.bn1(x_t_a))
        x_t_a = F.max_pool1d(x_t_a, kernel_pool)
        x_t_a = x_t_a.view(x_t_a.shape[0], -1)
        x_t_a = F.relu(self.fc_conv_a(x_t_a))
        
        x_t_b = self.conv1_b(x_t_b)
        x_t_b = self.conv2_b(x_t_b)
        x_t_b = F.relu(self.bn1(x_t_b))
        x_t_b = F.max_pool1d(x_t_b, kernel_pool)
        x_t_b = x_t_b.view(x_t_b.shape[0], -1)
        x_t_b = F.relu(self.fc_conv_b(x_t_b))
        
        
        x_cat = torch.cat((x_t_a, x_t_b, x), 1)
        
        x_cat = F.relu(self.fc1(x_cat))
        x_cat = self.dropout(x_cat)
        x_cat = F.relu(self.fc2(x_cat))
        x_cat = self.dropout(x_cat)
        output = F.relu(self.fc3(x_cat))
       
        return output


class Network_3(nn.Module):
    def __init__(self, x_dim, x_t_dim, hidden_dim=32):
        super(Network_3, self).__init__()
        
        self.conv1_a = nn.Conv1d(1, 6, kernel_size)
        self.conv2_a = nn.Conv1d(6, 12, kernel_size)
        self.fc_conv_a = nn.Linear(12, 1)
        
        self.conv1_b = nn.Conv1d(1, 6, kernel_size)
        self.conv2_b = nn.Conv1d(6, 12, kernel_size)
        self.fc_conv_b = nn.Linear(12, 1)
        
        self.conv1_c = nn.Conv1d(1, 6, kernel_size)
        self.conv2_c = nn.Conv1d(6, 12, kernel_size)
        self.fc_conv_c = nn.Linear(12, 1)
        
        
        concat_size = x_dim+x_t_dim*3
        self.fc1 = nn.Linear(concat_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1) 

        self.dropout = nn.Dropout(p=0.33)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(12)
        self.full_bn = nn.BatchNorm1d(concat_size)
    
    def forward(self, x, x_t_a, x_t_b, x_t_c):
        
        x_t_a = self.conv1_a(x_t_a)
        #x_t_a = self.relu(x_t_a)
        x_t_a = self.conv2_a(x_t_a)
        x_t_a = F.relu(self.bn1(x_t_a))
        x_t_a = F.max_pool1d(x_t_a, kernel_pool)
        x_t_a = x_t_a.view(x_t_a.shape[0], -1)
        x_t_a = F.relu(self.fc_conv_a(x_t_a))
        
        x_t_b = self.conv1_b(x_t_b)
        #x_t_b = self.relu(x_t_b)
        x_t_b = self.conv2_b(x_t_b)
        x_t_b = F.relu(self.bn1(x_t_b))
        x_t_b = F.max_pool1d(x_t_b, kernel_pool)
        x_t_b = x_t_b.view(x_t_b.shape[0], -1)
        x_t_b = F.relu(self.fc_conv_b(x_t_b))
        
        x_t_c = self.conv1_c(x_t_c)       
        #x_t_c = self.relu(x_t_c)
        x_t_c = self.conv2_c(x_t_c)
        x_t_c = F.relu(self.bn1(x_t_c))
        x_t_c = F.max_pool1d(x_t_c, kernel_pool)
        x_t_c = x_t_c.view(x_t_c.shape[0], -1)
        x_t_c = F.relu(self.fc_conv_c(x_t_c))
        
        
        x_cat = torch.cat((x_t_a, x_t_b, x_t_c, x), 1)
        
        x_cat = self.full_bn(x_cat)
        
        x_cat = F.relu(self.fc1(x_cat))
        x_cat = self.dropout(x_cat)
        x_cat = F.relu(self.fc2(x_cat))
        x_cat = self.dropout(x_cat)
        output = F.relu(self.fc3(x_cat))
       
        return output

class Network_4(nn.Module):
    def __init__(self, x_dim, x_t_dim, hidden_dim=128):
        super(Network_4, self).__init__()
        
        self.conv1 = nn.Conv1d(x_t_dim, 12, kernel_size)
        self.conv2 = nn.Conv1d(12, 32, kernel_size)
        self.fc_conv = nn.Linear(32, 16)
        
        concat_size = x_dim+16
        self.fc1 = nn.Linear(concat_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc3 = nn.Linear(hidden_dim//2, 1) 

        self.dropout = nn.Dropout(p=0.33)
        self.relu = nn.ReLU()
        self.full_bn = nn.BatchNorm1d(concat_size)
        self.bn_before_conv = nn.BatchNorm1d(x_t_dim)
    
    def forward(self, x, x_t):
        
        x_t = self.bn_before_conv(x_t)

        x_t = F.relu(self.conv1(x_t))
        x_t = F.relu(self.conv2(x_t))
        x_t = F.max_pool1d(x_t, kernel_pool)
        x_t = x_t.view(x_t.shape[0], -1)
        x_t = F.relu(self.fc_conv(x_t))
        
        x_cat = torch.cat((x_t, x), 1)
        
        x_cat = self.full_bn(x_cat)
        
        x_cat = F.relu(self.fc1(x_cat))
        x_cat = self.dropout(x_cat)
        x_cat = F.relu(self.fc2(x_cat))
        x_cat = self.dropout(x_cat)
        output = F.relu(self.fc3(x_cat))
       
        return output

class Network_lstm(nn.Module):
    def __init__(self, x_dim, x_t_dim, h_cell, num_lstm, hidden_dim=128):
        super(Network_lstm, self).__init__()
       
        
    
        self.lstm = nn.LSTM(input_size = x_t_dim[1], hidden_size = h_cell, 
                            num_layers = num_lstm, dropout = 0.2)
        self.fc_lstm = nn.Linear(h_cell*x_t_dim[2], hidden_dim//8)
        
        concat_size = hidden_dim//8 + x_dim[1]
        self.fc1 = nn.Linear(concat_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc3 = nn.Linear(hidden_dim//2, 1) 

        
        self.dropout = nn.Dropout(p=0.33)
        self.relu = nn.ReLU()
        self.full_bn = nn.BatchNorm1d(concat_size)
        self.bn_before_conv = nn.BatchNorm1d(x_t_dim[1])
        self._hidden_lstm = h_cell
        self.flatten = nn.Flatten()
        self._num_layers = num_lstm
    
    def forward(self, x, x_t):
        
        x_t = self.bn_before_conv(x_t)
        x_t = x_t.permute(2,0,1)
        
        h0 = torch.randn(self._num_layers, x_t.shape[1], self._hidden_lstm)
        c0 = torch.randn(self._num_layers, x_t.shape[1], self._hidden_lstm)
    
        x_t = self.lstm(x_t, (h0, c0))[0]
        
        x_t = x_t.permute(1,2,0)
        x_t = self.flatten(x_t)       
        x_t = self.fc_lstm(x_t)        
        
        x_cat = torch.cat((x_t, x), 1)
        
        x_cat = self.full_bn(x_cat)
        
        x_cat = F.relu(self.fc1(x_cat))
        x_cat = self.dropout(x_cat)
        x_cat = F.relu(self.fc2(x_cat))
        x_cat = self.dropout(x_cat)
        output = F.relu(self.fc3(x_cat))
       
        return output
