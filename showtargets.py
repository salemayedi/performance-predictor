import numpy as np
import matplotlib.pyplot as plt
import torch


train_temporal_data = torch.load('data/train_temporal_data.pt')
train_data_tensor = torch.load('data/train_data_tensor.pt')
train_targets = torch.load('data/train_targets.pt')
val_temporal_data = torch.load('data/val_temporal_data.pt')
val_data_tensor = torch.load('data/val_data_tensor.pt')
val_targets = torch.load('data/val_targets.pt')
test_temporal_data = torch.load('data/test_temporal_data.pt')
test_data_tensor = torch.load('data/test_data_tensor.pt')
test_targets = torch.load('data/test_targets.pt')


test_t = test_targets.detach().numpy()
train_t = train_targets.detach().numpy()
val_t = val_targets.detach().numpy()

bins = 250
plt.hist(train_t, bins, label="train")
plt.hist(test_t, bins, label="test")
plt.hist(val_t, bins, label="val")
plt.legend()
plt.grid()
plt.xlabel("Validation_accuracy")
plt.ylabel("Frequency")
plt.show()

