from api import Benchmark
import numpy as np


def cut_data(data, cut_position):
    targets = []
    for dp in data:
        targets.append(dp["Train/val_accuracy"][50])
        for tag in dp:
            if tag.startswith("Train/"):
                dp[tag] = dp[tag][0:cut_position]
    return data, targets

def read_data(bench_dir, dataset_name):
    bench = Benchmark(bench_dir, cache=False)
    dataset_name = 'Fashion-MNIST'
    n_configs = bench.get_number_of_configs(dataset_name)
    # Query API
    data = []
    for config_id in range(n_configs):
        data_point = dict()
        data_point["config"] = bench.query(dataset_name=dataset_name, tag="config", config_id=config_id)
        for tag in bench.get_queriable_tags(dataset_name=dataset_name, config_id=config_id):
            if tag.startswith("Train/"):
                data_point[tag] = bench.query(dataset_name=dataset_name, tag=tag, config_id=config_id)    
        data.append(data_point)
        
    # Split: 50% train, 25% validation, 25% test (the data is already shuffled)
    indices = np.arange(n_configs)
    ind_train = indices[0:int(np.floor(0.5*n_configs))]
    ind_val = indices[int(np.floor(0.5*n_configs)):int(np.floor(0.75*n_configs))]
    ind_test = indices[int(np.floor(0.75*n_configs)):]

    array_data = np.array(data)
    train_data = array_data[ind_train]
    val_data = array_data[ind_val]
    test_data = array_data[ind_test]
    
    # Cut curves for validation and test
    cut_position = 11
    val_data, val_targets = cut_data(val_data, cut_position)
    test_data, test_targets = cut_data(test_data, cut_position)
    train_data, train_targets = cut_data(train_data, 51)   # Cut last value as it is repeated
    
    return train_data, val_data, test_data, train_targets, val_targets, test_targets


def remove_uninformative_features(train_data): 
    store_config = {}
    informative_config = {}
    uninformative_config = {}

    for indx in range(len(train_data)):
        for value, key in train_data[indx]["config"].items():
            store_config[value] = []
    print("Seeking for uninformative data...")
    print()

    for value in store_config:
        for i in range(len(train_data)):
            for value_inner, key in train_data[i]["config"].items():
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
    print("Removing uninformative parameters from dataset...")        
    train_data_clean = train_data.copy()
    for i in range(len(train_data)):
        for value, key in uninformative_config.items():
            train_data_clean[i]["config"].pop(value)
            
    return train_data_clean
