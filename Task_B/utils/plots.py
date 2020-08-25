import matplotlib.pyplot as plt
import numpy as np

def plot_evolution (train_hist, val_hist, path):
    x = np.linspace(0, len(train_hist)*50, len(train_hist))
    plt.plot(x, train_hist, label='train')
    plt.plot(x, val_hist, label='val')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('mse loss')
    plt.ylim(-10, 60)
    plt.grid()
    plt.savefig(path)
    plt.show()