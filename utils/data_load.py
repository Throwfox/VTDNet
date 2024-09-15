import torch
from torch.utils.data import DataLoader,Dataset
from sklearn.model_selection import ShuffleSplit
import numpy as np

class CustomDataset_syn(Dataset):
    def __init__(self, dataset):
        #max length 30
        self.x = torch.from_numpy(dataset['covariates']).to(torch.float32)
        self.treatment = torch.from_numpy(dataset['treatments']).to(torch.float32)
        self.outcome = torch.from_numpy(dataset['outcomes']).to(torch.float32)
        self.delta_t = torch.ones(self.x.shape[0],self.x.shape[1], 1) # delta is consistent across this dataset
        self.varing_length = torch.from_numpy(dataset['sequence_length'])
        self.cf1 = torch.from_numpy(dataset['cf_outcomes_t1']).to(torch.float32)
        self.cf2 = torch.from_numpy(dataset['cf_outcomes_t2']).to(torch.float32)
        self.cf3 = torch.from_numpy(dataset['cf_outcomes_t3']).to(torch.float32)
        assert len(self.x) == len(self.treatment) == len(self.outcome)==len(self.varing_length), "Mismatch in dataset sizes"
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        x = self.x[idx]
        treatment = self.treatment[idx]
        outcome = self.outcome[idx]
        delta_t = self.delta_t[idx]  # Assuming delta_t is a single consistent value
        varing_length = self.varing_length[idx]
        cf1 = self.cf1[idx]
        cf2 = self.cf1[idx]
        cf3 = self.cf1[idx]
        return x, delta_t, treatment, outcome, varing_length, cf1, cf2, cf3

def get_syndata_splits(dataset, train_index, val_index, test_index):
    dataset_keys = ['covariates', 'confounders', 'treatments', 
                    'outcomes', 'cf_outcomes_t1', 'cf_outcomes_t2', 'cf_outcomes_t3']

    dataset_train = dict()
    dataset_val = dict()
    dataset_test = dict()
    for key in dataset_keys:
        dataset_train[key] = dataset[key][train_index, :, :]
        dataset_val[key] = dataset[key][val_index, :, :]
        dataset_test[key] = dataset[key][test_index, :, :]

    _, length, num_covariates = dataset_train['covariates'].shape

    key = 'sequence_length'
    dataset_train[key] = dataset[key][train_index]
    dataset_val[key] = dataset[key][val_index]
    dataset_test[key] = dataset[key][test_index]

    return dataset_train, dataset_val, dataset_test

def loader_all(dataset, task, batch_size):

    dataset = np.load('./Data/syn_data/syn_data_gamma_'+str(task)+'.npy', allow_pickle=True).item()        
    #7,1,2
    shuffle_split = ShuffleSplit(n_splits=1, test_size=0.2, random_state=10)
    train_index, test_index = next(shuffle_split.split(dataset['covariates'][:, :, 0]))
    shuffle_split = ShuffleSplit(n_splits=1, test_size=0.125, random_state=10)
    train_index, val_index = next(shuffle_split.split(dataset['covariates'][train_index, :, 0]))
    
    dataset_train, dataset_val, dataset_test = get_syndata_splits(dataset, train_index, val_index, test_index)
    
    train_dataset = CustomDataset_syn(dataset_train)
    val_dataset = CustomDataset_syn(dataset_val)
    test_dataset = CustomDataset_syn(dataset_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
    return train_loader, val_loader, test_loader