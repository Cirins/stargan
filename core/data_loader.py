import numpy as np
import torch
from torch.utils import data
import pickle
from torch.utils.data import WeightedRandomSampler



class train_dataset(data.Dataset):
    def __init__(self, dataset, class_names, num_df_domains, finetune=False):
        
        # Load the dataset
        with open(f'data/{dataset}.pkl', 'rb') as f:
            x, y, k = pickle.load(f)

        if finetune:
            print('Finetuning...')
            
            with open(f'data/{dataset}_fs.pkl', 'rb') as f:
                fs = pickle.load(f)
            
            # Filter out the samples that are not used for finetuning
            x = x[fs == 1]
            y = y[fs == 1]
            k = k[fs == 1]
        
            # Define train dataset
            x_train = x[k >= num_df_domains]
            y_train = y[k >= num_df_domains]
            k_train = k[k >= num_df_domains] - num_df_domains

        else:
            # Define train dataset
            x_train = x[k < num_df_domains]
            y_train = y[k < num_df_domains]
            k_train = k[k < num_df_domains]

        self.X_train = x_train.astype(np.float32)
        self.y_train = y_train.astype(np.int64)
        self.k_train = k_train.astype(np.int64)

        print(f'X_train shape is {self.X_train.shape}')

        classes_dict = {i: clss for i, clss in enumerate(class_names)}
        for i in range(len(class_names)):
            print(f'Number of {classes_dict[i]} samples: {len(y_train[y_train == i])}')

    def __len__(self):
        return len(self.y_train)
    
    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx], self.k_train[idx]



class test_dataset(data.Dataset):
    def __init__(self, dataset, class_names, num_df_domains):

        # Load the dataset
        with open(f'data/{dataset}.pkl', 'rb') as f:
            x, y, k = pickle.load(f)
        
        # Define test dataset
        x_test = x[k >= num_df_domains]
        y_test = y[k >= num_df_domains]
        k_test = k[k >= num_df_domains]

        self.X_test = x_test.astype(np.float32)
        self.y_test = y_test.astype(np.int64)
        self.k_test = k_test.astype(np.int64)

        print(f'X_test shape is {self.X_test.shape}')
        
        classes_dict = {i: clss for i, clss in enumerate(class_names)}
        for i in range(len(class_names)):
            print(f'Number of {classes_dict[i]} samples: {len(y_test[y_test == i])}')

    def __len__(self):
        return len(self.y_test)
    
    def __getitem__(self, idx):
        return self.X_test[idx], self.y_test[idx], self.k_test[idx]



class td_dataset(data.Dataset):
    def __init__(self, dataset, class_names, num_df_domains, finetune=False):
        
        # Load the dataset
        with open(f'data/splits/{dataset}_dp_map.pkl', 'rb') as f:
            x, y, k = pickle.load(f)

        if finetune:
            raise NotImplementedError

        self.X_td = x.astype(np.float32)
        self.y_td = y.astype(np.int64)
        print(f'X_td shape is {self.X_td.shape}')

        classes_dict = {i: clss for i, clss in enumerate(class_names)}
        for i in range(len(class_names)):
            print(f'Number of {classes_dict[i]} samples: {len(y[y == i])}')

    def __len__(self):
        return len(self.y_td)
    
    def __getitem__(self, idx):
        return self.X_td[idx], self.y_td[idx]



def get_class_weights(y):
    class_sample_count = np.bincount(y)
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in y])
    return samples_weight



def get_dataloaders(dataset, class_names, num_df_domains, batch_size, num_workers=2, finetune=False):
    """Create dataloaders for training and testing."""
    train_dataset_ = train_dataset(dataset, class_names, num_df_domains, finetune)
    test_dataset_ = test_dataset(dataset, class_names, num_df_domains)
    td_dataset_ = td_dataset(dataset, class_names, num_df_domains)

    train_loader = data.DataLoader(train_dataset_, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    test_loader = data.DataLoader(test_dataset_, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
    td_loader = data.DataLoader(td_dataset_, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    return train_loader, test_loader, td_loader



# def get_dataloaders(dataset, class_names, num_df_domains, batch_size, num_workers=2, finetune=False):
#     """Create dataloaders for training and testing."""
#     train_dataset_ = train_dataset(dataset, class_names, num_df_domains, finetune)
#     test_dataset_ = test_dataset(dataset, class_names, num_df_domains)
#     td_dataset_ = td_dataset(dataset, class_names, num_df_domains)

#     train_weights = get_class_weights(train_dataset_.y_train)
#     train_sampler = WeightedRandomSampler(train_weights, len(train_weights))

#     td_weights = get_class_weights(td_dataset_.y_td)
#     td_sampler = WeightedRandomSampler(td_weights, len(td_weights))

#     train_loader = data.DataLoader(train_dataset_, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, drop_last=True)
#     test_loader = data.DataLoader(test_dataset_, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
#     td_loader = data.DataLoader(td_dataset_, batch_size=batch_size, sampler=td_sampler, num_workers=num_workers, drop_last=True)

#     return train_loader, test_loader, td_loader
