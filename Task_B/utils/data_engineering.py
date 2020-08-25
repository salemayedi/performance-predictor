from api import Benchmark
import numpy as np
from sklearn.model_selection import train_test_split


def read_data(bench_dir, datasets):
    
    bench = Benchmark(bench_dir, cache=False)
    n_configs = bench.get_number_of_configs(datasets[0])
    data = [bench.query(dataset_name=d, tag="Train/val_accuracy", config_id=ind) for d in datasets for ind in range(n_configs)]
    configs = [bench.query(dataset_name=d, tag="config", config_id=ind) for d in datasets for ind in range(n_configs)]
    dataset_names = [d for d in datasets for ind in range(n_configs)]
    
    y = np.array([curve[-1] for curve in data])
    return np.array(configs), y, np.array(dataset_names)

class TrainValSplitter():
    """Splits 25 % data as a validation split."""
    
    def __init__(self, X, dataset_names):
        self.ind_train, self.ind_val = train_test_split(np.arange(len(X)), test_size=0.25, stratify=dataset_names)
        
    def split(self, a):
        return a[self.ind_train], a[self.ind_val]


def remove_uninformative_features(train_data): 
    store_config = {}
    informative_config = {}
    uninformative_config = {}

    for indx in range(len(train_data)):
        for value, key in train_data[indx].items():
            store_config[value] = []
    print("Seeking for uninformative data...")
    print()

    for value in store_config:
        for i in range(len(train_data)):
            for value_inner, key in train_data[i].items():
                store_config[value_inner].append(key)

        nTemp = store_config[value][0]
        bEqual = True
        for item in store_config[value]:
            if nTemp != item:
                bEqual = False
                break;
        if bEqual:
            print("All elements in list "+value+" are EQUAL")
            uninformative_config[value] = store_config[value]
        else:
            print("All elements in list "+value+" are different")
            informative_config[value] = store_config[value]

    print()
    print("Only "+str(len(informative_config))+"/"+str(len(store_config))+" parameters are informative.")
    print("Removing uninformative parameters from dataset...\n")        
    train_data_clean = train_data.copy()
    for i in range(len(train_data)):
        for value, key in uninformative_config.items():
            train_data_clean[i].pop(value)
            
    return train_data_clean
