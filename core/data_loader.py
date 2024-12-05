import numpy as np
import torch
from torch.utils import data
import pickle



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



def get_dataloaders(dataset, class_names, num_df_domains, batch_size, num_workers=2, finetune=False):
    """Create dataloaders for training and testing."""
    train_dataset_ = train_dataset(dataset, class_names, num_df_domains, finetune)
    test_dataset_ = test_dataset(dataset, class_names, num_df_domains)

    train_loader = data.DataLoader(train_dataset_, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = data.DataLoader(test_dataset_, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader
