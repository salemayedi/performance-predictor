# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class Network_1(nn.Module):
    def __init__(self, config_dim, meta_dim, hidden_dim=32):
        super(Network_1, self).__init__()
        
        self.fc1_m = nn.Linear(meta_dim, hidden_dim*2)
        self.fc2_m = nn.Linear(hidden_dim*2, hidden_dim)
        self.fc3_m = nn.Linear(hidden_dim, hidden_dim//8)
       
       
        self.fc1_c = nn.Linear(config_dim, hidden_dim)
        self.fc2_c = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_c = nn.Linear(hidden_dim, hidden_dim//8)
        
        
        self.fc4 = nn.Linear(hidden_dim//4, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim//4)
        self.fc6 = nn.Linear(hidden_dim//4, 1)

        self.dropout = nn.Dropout(p=0.)
        self.bn_config = nn.BatchNorm1d(config_dim)
        self.bn_meta = nn.BatchNorm1d(meta_dim)
        self.bn_cat = nn.BatchNorm1d(hidden_dim//4)
        self.sig = nn.Sigmoid()
        
    
    def forward(self, config, meta):
        
        
        meta = self.bn_meta(meta)        
        meta = F.relu(self.fc1_m(meta))
        meta = F.relu(self.fc2_m(meta))
        meta = F.relu(self.fc3_m(meta))
        
        config = F.relu(self.fc1_c(config))      
        config = F.relu(self.fc2_c(config))
        #config = self.dropout(config)
        config = F.relu(self.fc3_c(config))
           
        x = torch.cat((meta, config), 1)
        x = self.bn_cat(x)
        
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        output = self.fc6(x)
        
        return output
    
class Network_2(nn.Module):
    def __init__(self, config_dim, meta_dim, hidden_dim=32):
        super(Network_2, self).__init__()
        
        self.fc1_m = nn.Linear(meta_dim, 1)
       
        self.fc1_c = nn.Linear(config_dim, 1)

        self.fc2 = nn.Linear(2, 1)

        self.dropout = nn.Dropout(p=0.)
        self.bn_config = nn.BatchNorm1d(config_dim)
        self.bn_meta = nn.BatchNorm1d(meta_dim)
        self.bn_cat = nn.BatchNorm1d(2)
        self.sig = nn.Sigmoid()
        
    
    def forward(self, config, meta):
        
        
        meta = self.bn_meta(meta)        
        meta = F.relu(self.fc1_m(meta))
        config = self.bn_config(config)
        config = F.relu(self.fc1_c(config))
        x = torch.cat((meta, config), 1)
        x = self.bn_cat(x)
        output = self.fc2(x)
        
        return output
    
class regressor_classifier(nn.Module):
    def __init__(self, config_dim, meta_dim, hidden_dim=32):
        super(regressor_classifier, self).__init__()
        
        self.fc1_classifier = nn.Linear(meta_dim-1, 4)
        self.fc2_classifier = nn.Linear(hidden_dim, 4)
       
        self.fc1_c = nn.Linear(config_dim, hidden_dim)
        self.fc2_c = nn.Linear(hidden_dim, hidden_dim//4)
        self.fc3_c = nn.Linear(hidden_dim, hidden_dim//8)
        
        
        self.fc4 = nn.Linear(hidden_dim//8, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim//4)
        self.fc6 = nn.Linear(hidden_dim//4, 4)
        self.fc7 = nn.Linear(4, 1)

        self.dropout = nn.Dropout(p=0.2)
        self.bn_config = nn.BatchNorm1d(config_dim)
        self.bn_meta = nn.BatchNorm1d(meta_dim-1)
        self.bn_cat = nn.BatchNorm1d(hidden_dim//4)
        self.sig = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        
    
    def forward(self, config, meta):
        
        #print(meta.shape)
        
        meta = self.bn_meta(meta)
        output_class = self.fc1_classifier(meta)
        #output_class = self.fc2_classifier(meta)
        
        #print(output_class)
        output_class = self.softmax(output_class)
        
        config = self.bn_config(config)
        config = F.relu(self.fc1_c(config))   
        config = self.dropout(config)
        config = F.relu(self.fc2_c(config))
        config = self.dropout(config)
        output_multireg = self.fc6(config)
        
        otuput_reg =  self.fc7(output_class*output_multireg)
        
        return otuput_reg, output_class