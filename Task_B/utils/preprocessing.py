import torch
import numpy as np
import pandas as pd

def DictToTensor (X):
    dim_config = len(X[0])
    data_torch = torch.zeros(len(X), dim_config)
    for i in range(len(X)):
        j = 0
        for _, key in X[i].items():
            data_torch[i,j] = key
            j+=1
    return data_torch

def extractDataAsTensor(x):
    data = []
    for i in range(len(x)):
        data.append(x[i])
    return torch.FloatTensor(data)

def getmetadata(x, names, metadata):
    
    for i in range(len(x)):
        name = names[i]
        vec = np.zeros((len(x), 62))
        j = 0
        for key, val in metadata[name].items():
            vec[i, j] = val
            if j>60:
                break
            j +=1
    return vec
            
def cleanNan(x):
    df = pd.DataFrame(x)
    df = df.dropna(axis='columns')
    return df.values