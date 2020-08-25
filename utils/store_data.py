from api import Benchmark
from utils.data_engineering import read_data, remove_uninformative_features
from utils.preprocessing import DictToTensor, extractDataAsTensor
import torch


#Loading the data set
bench_dir = "./../data/fashion_mnist.json"
dataset_name = 'Fashion-MNIST'

train_data, val_data, test_data, train_targets, val_targets, test_targets = read_data(bench_dir, dataset_name)

train_data_clean = remove_uninformative_features(train_data)
val_data_clean = remove_uninformative_features(val_data)
test_data_clean = remove_uninformative_features(test_data)


train_loss = extractDataAsTensor(train_data, "Train/loss").unsqueeze(1)
train_acc = extractDataAsTensor(train_data, "Train/train_accuracy").unsqueeze(1)
train_val_acc=extractDataAsTensor(train_data, "Train/val_accuracy").unsqueeze(1)
train_grad = extractDataAsTensor(train_data, "Train/gradient_norm").unsqueeze(1)


val_loss = extractDataAsTensor(val_data, "Train/loss").unsqueeze(1)
val_acc = extractDataAsTensor(val_data, "Train/train_accuracy").unsqueeze(1)
val_val_acc = extractDataAsTensor(val_data, "Train/val_accuracy").unsqueeze(1)
val_grad = extractDataAsTensor(val_data, "Train/gradient_norm").unsqueeze(1)



test_loss = extractDataAsTensor(test_data_clean, "Train/loss").unsqueeze(1)
test_acc = extractDataAsTensor(test_data_clean, "Train/train_accuracy").unsqueeze(1)
test_val_acc = extractDataAsTensor(test_data_clean, "Train/val_accuracy").unsqueeze(1)
test_grad = extractDataAsTensor(test_data_clean, "Train/gradient_norm").unsqueeze(1)



train_temporal_data = torch.cat((train_loss, train_acc, train_val_acc, train_grad), 1)
val_temporal_data = torch.cat((val_loss, val_acc, val_val_acc, val_grad), 1)
test_temporal_data = torch.cat((test_loss, test_acc, test_val_acc, test_grad), 1)

train_data_tensor = DictToTensor(train_data_clean, "config")
val_data_tensor = DictToTensor(val_data_clean, "config")
test_data_tensor = DictToTensor(test_data_clean, "config")

train_targets = torch.FloatTensor(train_targets).reshape(-1,1)
val_targets = torch.FloatTensor(val_targets).reshape(-1,1)
test_targets = torch.FloatTensor(test_targets).reshape(-1,1)


train_temporal_data = train_temporal_data[:,:,0:val_temporal_data.shape[-1]]

torch.save(train_temporal_data, './../data/train_temporal_data.pt') #and torch.load('file.pt')
torch.save(train_data_tensor, './../data/train_data_tensor.pt')
torch.save(train_targets, './../data/train_targets.pt')
torch.save(val_temporal_data, './../data/val_temporal_data.pt')
torch.save(val_data_tensor, './../data/val_data_tensor.pt')
torch.save(val_targets, './../data/val_targets.pt')
torch.save(test_temporal_data, './../data/test_temporal_data.pt')
torch.save(test_data_tensor, './../data/test_data_tensor.pt')
torch.save(test_targets, './../data/test_targets.pt')