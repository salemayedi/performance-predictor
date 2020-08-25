import torch
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


class Model():
    def __init__(self, model, max_epochs, btch_size, loss, optimizer, scheduler, step_lr, gamma, path, path_model= None):
        self._max_epochs = max_epochs
        self._batch_size = btch_size
        self._loss = loss
        self._opt = optimizer
        self._model = model
        self._scheduler = scheduler
        self._path = path
        if path_model != None:
            checkpoint = torch.load(path_model)
            self._model.load_state_dict(checkpoint['model_state_dict'])
            self._opt.load_state_dict(checkpoint['optimizer_state_dict'])
            self._epochs = checkpoint['epochs']

        
    def train(self, config_train, meta_train, y_train, config_val, meta_val, y_val):
        # Train the network
        print("\n START !")

        train_loss_history = []
        val_loss_history = []
        steps = config_train.shape[0]//self._batch_size
        running_loss = 0.0
        for epoch in range(1,self._max_epochs):
            for indx in range(steps):
                self._opt.zero_grad()
                outputs = self._model(config_train[indx*self._batch_size:(indx+1)*self._batch_size, :],
                                                   meta_train[indx*self._batch_size:(indx+1)*self._batch_size, :])
                                                        
                loss = self._loss(outputs, y_train[indx*self._batch_size:(indx+1)*self._batch_size, :])
                loss.backward()
                self._opt.step()
                self._scheduler.step(epoch + indx/steps) 
                #self._scheduler.step()
                running_loss += loss.item()
                if epoch % 5 == 0 and indx == 0:    # print every 1000 mini-batches
                    
                    train_loss_history.append(running_loss/(steps*50))
                    with torch.no_grad():
                        self._model.eval()
                        predicted = self._model(config_val, meta_val)
                        score = mean_squared_error(predicted, y_val)
                        val_loss_history.append(score)
                        
                    print('[%d / %5d] train_loss: %.3f val_loss: %.3f' %
                          (epoch , self._max_epochs, running_loss/(steps*50), score))
                    running_loss = 0.0
                if epoch % 100 == 0 :
                    torch.save({
                    'epochs': epoch,
                    'model_state_dict': self._model.state_dict(),
                    'optimizer_state_dict': self._opt.state_dict(),
                    }, self._path+str(epoch)+".pkl")
                    
        print('END ! \n')
        return train_loss_history, val_loss_history
    
    def test(self, test_data, test_temporal, test_targets):
        with torch.no_grad():
            self._model.eval()
            predicted = self._model(test_data, test_temporal)
            score = mean_squared_error(predicted, test_targets)
        print("The MSE on Test : ", score)
        plt.scatter(test_targets, predicted)
        plt.show()
        
        
        

        
