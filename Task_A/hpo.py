import pickle
import logging
from typing import Tuple, List
from sklearn.metrics import mean_squared_error
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from hpbandster.core.worker import Worker
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB
import torch
import torch.nn as nn

from models.dnn import Network_lstm

logging.getLogger('hpbandster').setLevel(logging.DEBUG)

from collections import OrderedDict

def evaluate_loss(model: nn.Module, x, x_t, target) -> float:
    model.eval()
    with torch.no_grad():
        output = model(x, x_t)
        score = mean_squared_error(output, target)
    return(score.item(0))


class PyTorchWorker(Worker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.train_temporal_data = torch.load('data/train_temporal_data.pt')
        self.train_data_tensor = torch.load('data/train_data_tensor.pt')
        self.train_targets = torch.load('data/train_targets.pt')
        self.val_temporal_data = torch.load('data/val_temporal_data.pt')
        self.val_data_tensor = torch.load('data/val_data_tensor.pt')
        self.val_targets = torch.load('data/val_targets.pt')
        self.test_temporal_data = torch.load('data/test_temporal_data.pt')
        self.test_data_tensor = torch.load('data/test_data_tensor.pt')
        self.test_targets = torch.load('data/test_targets.pt')

    #@staticmethod
    def get_lstm_model(self, params) -> nn.Module:
        net = Network_lstm(*params)
        return net

 
    #@staticmethod
    def get_configspace(self) -> CS.ConfigurationSpace:

        cs = CS.ConfigurationSpace()
        
        lr = CSH.UniformFloatHyperparameter(
            'lr', lower=1e-6, upper=1e-0, default_value=1e-2, log=True)
        
        #Number of h cells on LSTM
        h_cell = CSH.UniformIntegerHyperparameter(
            'h_cells', lower=2, upper=11, default_value=3)
        #Number of LSTM Concatenated
        num_lstm = CSH.UniformIntegerHyperparameter(
            'num_lstm', lower=1, upper=8, default_value=3)
        #Hidden Fully Conected layers dimention
        hidden_dim = CSH.UniformIntegerHyperparameter(
            'hidden_dim', lower=16, upper=512, default_value=128)
    
        cs.add_hyperparameters([h_cell, num_lstm, hidden_dim, lr])
    
        return cs

    def compute(self, config: CS.Configuration, budget: float, working_directory: str,
                *args, **kwargs) -> dict:
        """Evaluate a function with the given config and budget and return a loss.
        
        Bohb tries to minimize the returned loss.
        
        In our case the function is the training and validation of a model,
        the budget is the number of epochs and the loss is the validation error.
        """
        input_config_size = self.train_data_tensor.shape
        input_temporal_size = self.train_temporal_data.shape
        #print('!!!!!!!!!!!!!!CONFIGURATIONS :       ', config)
        parameters = (input_config_size, input_temporal_size, 
                      config['h_cells'], config['num_lstm'], config['hidden_dim'])
        
        #print("!!!!!!!!!!!!!!PARAMETERS  : ", parameters)
        model = self.get_lstm_model(parameters)
        
        # START TODO ################ (3points)
        batch_size = 64
        criterion = nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        
        # train the model for `epochs` and save the validation error for each epoch in
        # val_errors
        
        steps = self.train_data_tensor.shape[0]//batch_size
        for epoch in range(int(budget)):
            loss = 0.
            model.train()
            for indx in range(steps):
                optimizer.zero_grad()
                outputs = model(self.train_data_tensor[indx*batch_size:(indx+1)*batch_size, :], 
                                self.train_temporal_data[indx*batch_size:(indx+1)*batch_size, :])
                                                        
                loss = criterion(outputs, self.train_targets[indx*batch_size:(indx+1)*batch_size, :])
                loss.backward()
                optimizer.step()
                #scheduler.step(epoch + indx/steps) 
 
        train_loss = evaluate_loss(model, self.train_data_tensor, self.train_temporal_data, self.train_targets)
        validation_loss = evaluate_loss(model, self.val_data_tensor, self.val_temporal_data, self.val_targets)
        test_loss = evaluate_loss(model, self.test_data_tensor, self.test_temporal_data, self.test_targets)

        return ({
                'loss': validation_loss,  # remember: HpBandSter minimizes the loss!
                'info': {'test_loss': test_loss,
                         'train_loss': train_loss,
                         'valid_loss': validation_loss,
                         'model': str(model)}
                })

   
# RUN BOHB
import os
working_dir = "./results/BOHB/"
result_file = os.path.join(working_dir, 'bohb_result.pkl')
nic_name = 'lo0'
port = 0
run_id = 'bohb_run_1'
n_bohb_iterations = 10
min_budget = 500
max_budget = 3500

try:
    # Start a nameserver #####
    # get host
    try:
        host = hpns.nic_name_to_host(nic_name)
    except ValueError as e:
        host = "localhost"
        print(e)
        print("ValueError getting host from nic_name {}, "
              "setting to localhost.".format(nic_name))    
    
    ns = hpns.NameServer(run_id=run_id, host=host, port=port,
                         working_directory=working_dir)
    ns_host, ns_port = ns.start()
    print(ns_host)
    print()
    print(ns_port)

    # Start local worker
    w = PyTorchWorker(run_id=run_id, host=host, nameserver=ns_host,
                      nameserver_port=ns_port, timeout=120)
    w.run(background=True)
    print('m here!!!!!!!!!!!!!!!!!!!!')
    # Run an optimizer
    print(' GET CONFIG   : ',w.get_configspace)
    bohb = BOHB(configspace=w.get_configspace(),
                run_id=run_id,
                host=host,
                nameserver=ns_host,
                nameserver_port=ns_port,
                min_budget=min_budget, max_budget=max_budget)
    result = bohb.run(n_iterations=n_bohb_iterations)
    print('AFTER BOHB !!!!!!!!!!!!!!!!!!!!')

    print("Write result to file {}".format(result_file))
    with open(result_file, 'wb') as f:
        pickle.dump(result, f)
finally:
    bohb.shutdown(shutdown_workers=True)
    ns.shutdown()
    
#  load a saved result object if necessary
with open(result_file, 'rb') as f:
    result = pickle.load(f)
inc_id = result.get_incumbent_id()  # get config_id of incumbent (lowest loss)
inc_run = result.get_runs_by_id(inc_id)[-1]  # get run with this config_id on highest budget
best_error, best_model = inc_run.loss, inc_run.info['model']
print("The best model (config_id {}) has the lowest final error with {:.4f}."
      .format(inc_run.config_id, best_error))
print(best_model)

## alternative solution:
max_budget_runs = result.get_all_runs(only_largest_budget=True)
best_error, best_config_id, best_model = sorted((run.loss, run.config_id, run.info['model']) 
                                                for run in max_budget_runs)[0]
print("The best model (config_id {}) has the lowest final error with {:.4f}."
      .format(best_config_id, best_error))
print(best_model)
