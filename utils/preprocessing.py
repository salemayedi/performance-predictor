import torch

def DictToTensor (X, name_data):
    dim_config = len(X[0][name_data])
    data_torch = torch.zeros(len(X), dim_config)
    for i in range(len(X)):
        j = 0
        for _, key in X[i][name_data].items():
            data_torch[i,j] = key
            j+=1
    return data_torch

def extractDataAsTensor(x, name_data):
    data = []
    for i in range(len(x)):
        data.append(x[i][name_data])
    return torch.FloatTensor(data)
        