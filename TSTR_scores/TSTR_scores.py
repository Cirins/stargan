import pickle
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import LambdaLR, StepLR
import torch.optim as optim
import os
import csv
import random
import copy
import itertools
from copy import deepcopy

seed = 2710
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

config = {
    'realworld': {
        'dataset_name': 'realworld',
        'num_df_domains': 10,
        'num_dp_domains': 5,
        'num_classes': 4,
        'class_names': ['WAL', 'RUN', 'CLD', 'CLU'],
        'num_timesteps': 128,
        'num_channels': 3,
        'num_classes': 4,
    },
    'cwru': {
        'dataset_name': 'cwru_256_3ch_5cl',
        'num_df_domains': 4,
        'num_dp_domains': 4,
        'num_classes': 5,
        'class_names': ['IR', 'Ball', 'OR_centred', 'OR_orthogonal', 'OR_opposite'],
        'num_timesteps': 256,
        'num_channels': 3,
        'num_classes': 5,
    },
    'realworld_mobiact': {
        'dataset_name': 'realworld_mobiact',
        'num_df_domains': 15,
        'num_dp_domains': 61,
        'num_classes': 4,
        'class_names': ['WAL', 'RUN', 'CLD', 'CLU'],
        'num_timesteps': 128,
        'num_channels': 3,
        'num_classes': 4,
    },
    'mobiact_realworld': {
        'dataset_name': 'mobiact_realworld',
        'num_df_domains': 61,
        'num_dp_domains': 15,
        'num_classes': 4,
        'class_names': ['WAL', 'RUN', 'CLD', 'CLU'],
        'num_timesteps': 128,
        'num_channels': 3,
        'num_classes': 4,
    }
}

num_epochs = 40
patience = -1
num_runs = 10
print(f"Number of epochs: {num_epochs} | Patience: {patience} | Number of runs: {num_runs}\n")


def get_data(dataset, domains_set, src_class, domain=None, rot=False):
    # Load configurations
    dataset_name = config[dataset]['dataset_name']
    class_idx = config[dataset]['class_names'].index(src_class)
    num_df_domains = config[dataset]['num_df_domains']

    if rot:
        dataset_name += '_rot'

    dataset_name += f'_{domains_set}'

    try:
        with open(f'data/{dataset_name}.pkl', 'rb') as f:
            x, y, k = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset files for {dataset} not found.")

    # Additional domain filtering if specified
    if domain is not None:
        domain_mask = (k == domain)
        x, y, k = x[domain_mask], y[domain_mask], k[domain_mask]

    assert len(x) > 0, f"No data found"
    assert len(x) == len(y) == len(k), f"Data length mismatch"

    return x, y, k



class TSTRFeatureExtractor(nn.Module):
    def __init__(self, num_timesteps=128, num_channels=3):
        super(TSTRFeatureExtractor, self).__init__()

        self.conv1 = nn.Conv1d(num_channels, 16, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn4 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)
        return x


class TSTRClassifier(nn.Module):
    def __init__(self, num_timesteps=128, num_channels=3, num_classes=5):
        super(TSTRClassifier, self).__init__()

        self.feature_extractor = TSTRFeatureExtractor(num_timesteps, num_channels)
        self.fc_shared = nn.Linear(num_timesteps * 8, 100)
        self.fc_class = nn.Linear(100, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.relu(self.fc_shared(x))
        class_outputs = self.fc_class(x)
        return class_outputs



def remap_labels(y):
    label_map = {clss: i for i, clss in enumerate(np.unique(y))}
    return np.array([label_map[clss] for clss in y])



def get_dataloader(x, y, shuffle=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.tensor(x, dtype=torch.float32, device=device)
    y = remap_labels(y)
    y = torch.tensor(y, dtype=torch.long, device=device)

    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=shuffle)

    return loader


def random_rotation_matrix():
    """Generate a random rotation matrix from a predefined set of quaternions."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Randomly generate a quaternion
    q = np.random.rand(4)

    # Convert quaternion to rotation matrix
    q = torch.tensor(q, device=device, dtype=torch.float32)
    q = q / torch.norm(q)  # Normalize quaternion
    q0, q1, q2, q3 = q

    R = torch.tensor([
        [1 - 2*q2**2 - 2*q3**2, 2*q1*q2 - 2*q3*q0, 2*q1*q3 + 2*q2*q0],
        [2*q1*q2 + 2*q3*q0, 1 - 2*q1**2 - 2*q3**2, 2*q2*q3 - 2*q1*q0],
        [2*q1*q3 - 2*q2*q0, 2*q2*q3 + 2*q1*q0, 1 - 2*q1**2 - 2*q2**2]
    ], device=device, dtype=torch.float32)

    return R, q


def augment_batch(x_real):
    """Apply random rotation to the batch of real time series."""
    min_val, max_val = -19.61, 19.61
    x_real = x_real * (max_val - min_val) + min_val  # De-normalize
    R, q = random_rotation_matrix()
    x_real = torch.matmul(R, x_real)  # Apply rotation
    x_real = (x_real - min_val) / (max_val - min_val)  # Re-normalize
    return x_real, q



class CORAL(nn.Module):
    def __init__(self):
        super(CORAL, self).__init__()

    def forward(self, source, target):
        d = source.size(1)

        # source covariance
        xm = torch.mean(source, 0, keepdim=True) - source
        xc = xm.t() @ xm

        # target covariance
        xmt = torch.mean(target, 0, keepdim=True) - target
        xct = xmt.t() @ xmt

        # frobenius norm between source and target
        loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
        loss = loss / (4 * d * d)
        return loss



def evaluate_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            loss = loss_fn(outputs, y_batch)
            total_loss += loss.item()

            _, predicted_labels = torch.max(outputs, 1)
            all_preds.extend(predicted_labels.detach().cpu().numpy())
            all_labels.extend(y_batch.detach().cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    total_loss /= len(test_loader)

    return accuracy, total_loss, f1



def train_model(model, train_loader, val_loader, optimizer, num_epochs=100, augment=False, patience=-1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()

    # loss_train = []
    # loss_val = []
    # accuracy_val = []

    # best_model_state = None
    # best_epoch = 0
    # best_loss = np.inf
    # best_accuracy = 0
    # best_f1 = 0
    # epochs_no_improve = 0

    # Set up linear learning rate decay
    # lambda_lr = lambda epoch: 1 - epoch / num_epochs
    # scheduler = LambdaLR(optimizer, lr_lambda=lambda_lr)
    # scheduler = StepLR(optimizer, step_size=50, gamma=0.1)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            if augment:
                x_batch, _ = augment_batch(x_batch)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        total_loss /= len(train_loader)
        # loss_train.append(total_loss)

        # # Update learning rate
        # scheduler.step()

        # val_accuracy, val_loss, val_f1 = evaluate_model(model, val_loader)
        # if val_loss < best_loss:
        #     best_epoch = epoch
        #     best_accuracy = val_accuracy
        #     best_f1 = val_f1
        #     best_loss = val_loss
        #     best_model_state = deepcopy(model.state_dict())
        #     epochs_no_improve = 0
        # else:
        #     epochs_no_improve += 1

        # loss_val.append(val_loss)
        # accuracy_val.append(val_accuracy)

        # current_lr = scheduler.get_last_lr()[0]

        if (epoch+1) % 5 == 0:
            # print(f"\tEpoch {epoch + 1}/{num_epochs} - Train loss: {total_loss:.4f} - Val loss: {val_loss:.4f} - Val accuracy: {val_accuracy:.4f} - Val F1: {val_f1:.4f} - LR: {current_lr:.2e}")
            print(f"\tEpoch {epoch + 1}/{num_epochs} - Train loss: {total_loss:.4f}")

        # # Early stopping
        # if patience > 0 and epochs_no_improve >= patience:
        #     print(f"Early stopping at epoch {epoch + 1}")
        #     break

    # print(f"\tBest epoch: {best_epoch + 1} - Best val loss: {best_loss:.4f} - Best val accuracy: {best_accuracy:.4f} - Best val F1: {best_f1:.4f}\n")

    # # Load best model state
    # model.load_state_dict(best_model_state)

    return model



def train_model_coral(model, train_loader, val_loader, coral_loader, optimizer, coral_weight=1e5, num_epochs=100, augment=False, patience=-1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    loss_coral = CORAL()

    # loss_train = []
    # loss_val = []
    # accuracy_val = []

    # best_model_state = None
    # best_epoch = 0
    # best_loss = np.inf
    # best_accuracy = 0
    # best_f1 = 0
    # epochs_no_improve = 0

    # Set up linear learning rate decay
    # lambda_lr = lambda epoch: 1 - epoch / num_epochs
    # scheduler = LambdaLR(optimizer, lr_lambda=lambda_lr)
    # scheduler = StepLR(optimizer, step_size=50, gamma=0.1)

    len_joint_loader = max(len(train_loader), len(coral_loader))

    print("Warning: 1 epoch")
    for epoch in range(1):
    
        if len(train_loader) > len(coral_loader):
            joint_loader = zip(train_loader, itertools.cycle(coral_loader))
        else:
            joint_loader = zip(itertools.cycle(train_loader), coral_loader)        
        
        model.train()
        total_loss = 0
        total_class_loss = 0
        total_coral_loss = 0
        for i, ((x_batch, y_batch), (x_coral_batch, _)) in enumerate(joint_loader):
            x_batch, y_batch, x_coral_batch = x_batch.to(device), y_batch.to(device), x_coral_batch.to(device)
            if augment:
                x_batch, _ = augment_batch(x_batch)
            optimizer.zero_grad()
            outputs = model(x_batch)
            class_loss = loss_fn(outputs, y_batch)
            coral_loss = coral_weight * loss_coral(model.feature_extractor(x_batch), model.feature_extractor(x_coral_batch))
            loss = class_loss + coral_loss
            if i % 50 == 0:
                print(f"Batch {i+1}/{len_joint_loader} - Class loss: {class_loss.item():.4f} - Coral loss: {coral_loss.item():.4f} - Total loss: {loss.item():.4f}")
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_class_loss += class_loss.item()
            total_coral_loss += coral_loss.item()
        total_loss /= len_joint_loader
        total_class_loss /= len_joint_loader
        total_coral_loss /= len_joint_loader
        # loss_train.append(total_loss)

        # # Update learning rate
        # scheduler.step()

        # val_accuracy, val_loss, val_f1 = evaluate_model(model, val_loader)
        # if val_loss < best_loss:
        #     best_epoch = epoch
        #     best_accuracy = val_accuracy
        #     best_f1 = val_f1
        #     best_loss = val_loss
        #     best_model_state = deepcopy(model.state_dict())
        #     epochs_no_improve = 0
        # else:
        #     epochs_no_improve += 1

        # loss_val.append(val_loss)
        # accuracy_val.append(val_accuracy)

        # current_lr = scheduler.get_last_lr()[0]

        if (epoch+1) % 5 == 0:
            # print(f"\tEpoch {epoch + 1}/{num_epochs} - Train loss: {total_loss:.4f} ({total_class_loss:.4f} + {total_coral_loss:.4f}) - Val loss: {val_loss:.4f} - Val accuracy: {val_accuracy:.4f} - Val F1: {val_f1:.4f} - LR: {current_lr:.2e}")
            print(f"\tEpoch {epoch + 1}/{num_epochs} - Train loss: {total_loss:.4f} ({total_class_loss:.4f} + {total_coral_loss:.4f})")

    #     # Early stopping
    #     if patience > 0 and epochs_no_improve >= patience:
    #         print(f"Early stopping at epoch {epoch + 1}")
    #         break

    # print(f"\tBest epoch: {best_epoch + 1} - Best val loss: {best_loss:.4f} - Best val accuracy: {best_accuracy:.4f} - Best val F1: {best_f1:.4f}\n")

    # # Load best model state
    # model.load_state_dict(best_model_state)

    return model



def train_only(x_train, y_train, dataset, num_epochs=100, augment=False, patience=-1):

    num_classes = len(np.unique(y_train))

    # x_tr, x_val, y_tr, y_val = train_test_split(x_train, y_train, test_size=0.2, stratify=y_train, shuffle=True, random_state=seed)

    # train_loader = get_dataloader(x_tr, y_tr, shuffle=True)
    # val_loader = get_dataloader(x_val, y_val)

    train_loader = get_dataloader(x_train, y_train, shuffle=True)
    val_loader = None
    print("Warning: no validation set")

    model = TSTRClassifier(num_timesteps=config[dataset]['num_timesteps'],
                           num_channels=config[dataset]['num_channels'],
                           num_classes=num_classes)
    initial_lr = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=1e-4)

    trained_model = train_model(model, train_loader, val_loader, optimizer, num_epochs=num_epochs, augment=augment, patience=patience)

    return trained_model



def train_and_test(x_train, y_train, x_test, y_test, dataset, num_epochs=100, augment=False, patience=-1, train_coral=False):
    assert np.array_equal(np.unique(y_train), np.unique(y_test)), f"Training and test labels do not match: {np.unique(y_train)} vs {np.unique(y_test)}"

    num_classes = len(np.unique(y_train))

    # x_tr, x_val, y_tr, y_val = train_test_split(x_train, y_train, test_size=0.2, stratify=y_train, shuffle=True, random_state=seed)

    if train_coral:
        # x_coral_train, x_test, y_coral_train, y_test = train_test_split(x_test, y_test, test_size=0.2, stratify=y_test, shuffle=True, random_state=seed)
        # print(f'x_coral_train.shape: {x_coral_train.shape} | np.unique(y_coral_train): {np.unique(y_coral_train)}')
        # print(f'x_coral_test.shape: {x_test.shape} | np.unique(y_coral_test): {np.unique(y_test)}')
        x_coral_train = x_test[y_test == 0]
        y_coral_train = y_test[y_test == 0]
        x_test = x_test[y_test != 0]
        y_test = y_test[y_test != 0]
        print(f'x_coral_train.shape: {x_coral_train.shape} | np.unique(y_coral_train): {np.unique(y_coral_train)}')
        print(f'x_test.shape: {x_test.shape} | np.unique(y_test): {np.unique(y_test)}')

    # train_loader = get_dataloader(x_tr, y_tr, shuffle=True)
    # val_loader = get_dataloader(x_val, y_val)
    # coral_loader = get_dataloader(x_coral_train, y_coral_train, shuffle=True)
    # test_loader = get_dataloader(x_test, y_test)

    train_loader = get_dataloader(x_train, y_train, shuffle=True)
    val_loader = None
    coral_loader = get_dataloader(x_test, y_test, shuffle=True)
    test_loader = get_dataloader(x_test, y_test)
    print("Warning: no validation set and coral set is the test set")

    model = TSTRClassifier(num_timesteps=config[dataset]['num_timesteps'],
                           num_channels=config[dataset]['num_channels'],
                           num_classes=num_classes)
    initial_lr = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=1e-4)

    if train_coral:
        trained_model = train_model_coral(model, train_loader, val_loader, coral_loader, optimizer, num_epochs=num_epochs, augment=augment, patience=patience)
    else:
        trained_model = train_model(model, train_loader, val_loader, optimizer, num_epochs=num_epochs, augment=augment, patience=patience)

    test_accuracy, test_loss, test_f1 = evaluate_model(trained_model, test_loader)

    return test_accuracy, test_loss, test_f1


def pseudo_labeling(model, x, y, dataset, num_epochs=100, augment=False, patience=-1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_pl, x_test, y_pl, y_test = train_test_split(x, remap_labels(y), test_size=0.2, stratify=y, shuffle=True, random_state=seed)
    print(f'x_pl.shape: {x_pl.shape} | np.unique(y_pl): {np.unique(y_pl)}')
    print(f'x_test.shape: {x_test.shape} | np.unique(y_test): {np.unique(y_test)}')

    x_pl = torch.tensor(x_pl, dtype=torch.float32, device=device)
    y_pl_pred = torch.argmax(model(x_pl), dim=1).detach().cpu().numpy()
    x_pl = x_pl.detach().cpu().numpy()

    return train_and_test(x_pl, y_pl_pred, x_test, y_test, dataset, num_epochs=num_epochs, augment=augment, patience=patience)

    



def train_classifier_cv(x_train, y_train, dataset, num_epochs=100):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    accs = []
    losses = []
    f1s = []

    for train_index, test_index in skf.split(x_train, y_train):
        x_train_fold, x_test_fold = x_train[train_index], x_train[test_index]
        y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]

        acc, loss, f1 = train_and_test(x_train_fold, y_train_fold, x_test_fold, y_test_fold, dataset, num_epochs)
        accs.append(acc)
        losses.append(loss)
        f1s.append(f1)
    return np.mean(accs), np.mean(losses), np.mean(f1s)



def fine_tune(model, x_train, y_train, num_epochs=100):
    raise NotImplementedError("Not updated for the new dataset format")
    # Freeze feature extraction layers
    for name, param in model.named_parameters():
        if 'conv' in name or 'bn' in name:
            param.requires_grad = False

    x_tr, x_val, y_tr, y_val = train_test_split(x_train, y_train, test_size=0.2, stratify=y_train, shuffle=True, random_state=seed)

    train_loader = get_dataloader(x_tr, y_tr, shuffle=True)
    val_loader = get_dataloader(x_val, y_val)

    initial_lr = 0.00001
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=initial_lr)

    trained_model = train_model(model, train_loader, val_loader, optimizer, num_epochs)

    return trained_model





def save_scores(source, domain, accuracy, loss, f1, name, dataset):
    results_dir = 'results'
    # Ensure the directory exists
    os.makedirs(results_dir, exist_ok=True)
    # Path to the CSV file
    file_path = os.path.join(results_dir, f'{dataset}_{name}.csv')
    # Check if the file exists
    file_exists = os.path.exists(file_path)

    # Open the file in append mode if it exists, or write mode if it doesn't
    with open(file_path, mode='a' if file_exists else 'w', newline='') as file:
        writer = csv.writer(file)
        # If the file does not exist, write the header
        if not file_exists:
            writer.writerow(['source', 'domain', 'accuracy', 'loss', 'f1'])
        # Write the data rows
        writer.writerow([source, domain, accuracy, loss, f1])




def compute_TSTR_Dp(dataset):
    accs = []
    f1s = []

    for src_class in config[dataset]['class_names']:
        if src_class != 'WAL':
            continue

        print(f"Source class: {src_class}\n")

        for domain in range(config[dataset]['num_df_domains'], config[dataset]['num_df_domains'] + config[dataset]['num_dp_domains']):
            print(f"Domain: {domain}")

            if domain == 74:
              continue

            # Load Dp data
            x_dp_dom, y_dp_dom, k_dp_dom = get_data(dataset, 'dp', src_class, domain)
            print(f'x_dp_dom.shape: {x_dp_dom.shape} | np.unique(y_dp_dom): {np.unique(y_dp_dom)} | np.unique(k_dp_dom): {np.unique(k_dp_dom)}\n')

            # Train and evaluate via cross-validation on Dp data
            print('Training and evaluating on Dp data via cross-validation...')
            acc, loss, f1 = train_classifier_cv(x_dp_dom, y_dp_dom, dataset, num_epochs=num_epochs)
            save_scores(src_class, domain, acc, loss, f1, 'Dp', dataset)
            accs.append(acc)
            f1s.append(f1)
            print(f'Source class: {src_class} | Domain: {domain} | Accuracy: {acc:.4f} | Loss: {loss:.4f} | F1: {f1:.4f}\n')

    print(f"Mean accuracy: {np.mean(accs):.4f} +- {np.std(accs):.4f}")
    print(f"Mean F1: {np.mean(f1s):.4f} +- {np.std(f1s):.4f}\n")




def compute_TSTR_Df(dataset):
    accs = []
    f1s = []

    for i in range(num_runs):

        accs_run = []
        f1s_run = []

        for src_class in config[dataset]['class_names']:
            if src_class != 'WAL':
                continue

            print(f"Source class: {src_class}\n")

            # Load Df data
            x_df, y_df, k_df = get_data(dataset, 'df', src_class)
            print(f'x_df.shape: {x_df.shape} | np.unique(y_df): {np.unique(y_df)} | np.unique(k_df): {np.unique(k_df)}\n')

            # Train on Df data
            print('Training on Df data...')
            df_model = train_only(x_df, y_df, dataset, num_epochs=num_epochs, patience=patience)

            for domain in range(config[dataset]['num_df_domains'], config[dataset]['num_df_domains'] + config[dataset]['num_dp_domains']):
                print(f"Domain: {domain}")

                # Load Dp data
                x_dp_dom, y_dp_dom, k_dp_dom = get_data(dataset, 'dp', src_class, domain)
                print(f'x_dp_dom.shape: {x_dp_dom.shape} | np.unique(y_dp_dom): {np.unique(y_dp_dom)} | np.unique(k_dp_dom): {np.unique(k_dp_dom)}')

                # Evaluate on Dp data
                acc, loss, f1 = evaluate_model(df_model, get_dataloader(x_dp_dom, y_dp_dom))
                save_scores(src_class, domain, acc, loss, f1, 'Df', dataset)
                accs_run.append(acc)
                f1s_run.append(f1)
                print(f'Source class: {src_class} | Domain: {domain} | Loss: {loss:.4f} | Accuracy: {acc:.4f} | F1: {f1:.4f}\n')

        print(f"Mean accuracy for run {i}: {np.mean(accs_run):.4f} +- {np.std(accs_run):.4f}")
        print(f"Mean F1 for run {i}: {np.mean(f1s_run):.4f} +- {np.std(f1s_run):.4f}\n")
        accs.append(np.mean(accs_run))
        f1s.append(np.mean(f1s_run))

    print(f"Mean accuracy over {num_runs} runs: {np.mean(accs):.4f} +- {np.std(accs):.4f}")
    print(f"Mean F1 over {num_runs} runs: {np.mean(f1s):.4f} +- {np.std(f1s):.4f}\n")




def compute_TSTR_Df_aug(dataset):
    accs = []
    f1s = []

    for i in range(num_runs):

        accs_run = []
        f1s_run = []

        for src_class in config[dataset]['class_names']:
            if src_class != 'WAL':
                continue

            print(f"Source class: {src_class}\n")

            # Load Df data
            x_df, y_df, k_df = get_data(dataset, 'df', src_class)
            print(f'x_df.shape: {x_df.shape} | np.unique(y_df): {np.unique(y_df)} | np.unique(k_df): {np.unique(k_df)}\n')

            # Train on Df data
            print('Training on Df data...')
            df_model = train_only(x_df, y_df, dataset, num_epochs=num_epochs, augment=True, patience=patience)

            for domain in range(config[dataset]['num_df_domains'], config[dataset]['num_df_domains'] + config[dataset]['num_dp_domains']):
                print(f"Domain: {domain}")

                # Load Dp data
                x_dp_dom, y_dp_dom, k_dp_dom = get_data(dataset, 'dp', src_class, domain)
                print(f'x_dp_dom.shape: {x_dp_dom.shape} | np.unique(y_dp_dom): {np.unique(y_dp_dom)} | np.unique(k_dp_dom): {np.unique(k_dp_dom)}')

                # Evaluate on Dp data
                acc, loss, f1 = evaluate_model(df_model, get_dataloader(x_dp_dom, y_dp_dom))
                save_scores(src_class, domain, acc, loss, f1, 'Df_aug', dataset)
                accs_run.append(acc)
                f1s_run.append(f1)
                print(f'Source class: {src_class} | Domain: {domain} | Loss: {loss:.4f} | Accuracy: {acc:.4f} | F1: {f1:.4f}\n')

        print(f"Mean accuracy for run {i}: {np.mean(accs_run):.4f} +- {np.std(accs_run):.4f}")
        print(f"Mean F1 for run {i}: {np.mean(f1s_run):.4f} +- {np.std(f1s_run):.4f}\n")
        accs.append(np.mean(accs_run))
        f1s.append(np.mean(f1s_run))

    print(f"Mean accuracy over {num_runs} runs: {np.mean(accs):.4f} +- {np.std(accs):.4f}")
    print(f"Mean F1 over {num_runs} runs: {np.mean(f1s):.4f} +- {np.std(f1s):.4f}\n")




def compute_TSTR_Syn(dataset, syn_name):
    accs = []
    f1s = []

    for i in range(num_runs):

        accs_run = []
        f1s_run = []

        for src_class in config[dataset]['class_names']:
            if src_class != 'WAL':
                continue

            print(f"Source class: {src_class}\n")

            for domain in range(config[dataset]['num_df_domains'], config[dataset]['num_df_domains'] + config[dataset]['num_dp_domains']):
                print(f"Domain: {domain}")

                # Load synthetic data
                x_syn_dom, y_syn_dom, k_syn_dom = get_data(dataset, syn_name, src_class, domain)
                print(f'x_syn_dom.shape: {x_syn_dom.shape} | np.unique(y_syn_dom): {np.unique(y_syn_dom)} | np.unique(k_syn_dom): {np.unique(k_syn_dom)}')

                # Load Dp data
                x_dp_dom, y_dp_dom, k_dp_dom = get_data(dataset, 'dp', src_class, domain)
                print(f'x_dp_dom.shape: {x_dp_dom.shape} | np.unique(y_dp_dom): {np.unique(y_dp_dom)} | np.unique(k_dp_dom): {np.unique(k_dp_dom)}\n')

                # Train on synthetic data and evaluate on Dp data
                print('Training on synthetic data...')
                acc, loss, f1 = train_and_test(x_syn_dom, y_syn_dom, x_dp_dom, y_dp_dom, dataset, num_epochs=num_epochs, patience=patience)
                save_scores(src_class, domain, acc, loss, f1, 'Syn', dataset)
                accs_run.append(acc)
                f1s_run.append(f1)
                print(f'Source class: {src_class} | Domain: {domain} | Loss: {loss:.4f} | Accuracy: {acc:.4f} | F1: {f1:.4f}\n')

        print(f"Mean accuracy for run {i}: {np.mean(accs_run):.4f} +- {np.std(accs_run):.4f}")
        print(f"Mean F1 for run {i}: {np.mean(f1s_run):.4f} +- {np.std(f1s_run):.4f}\n")
        accs.append(np.mean(accs_run))
        f1s.append(np.mean(f1s_run))

    print(f"Mean accuracy: {np.mean(accs):.4f} +- {np.std(accs):.4f}")
    print(f"Mean F1: {np.mean(f1s):.4f} +- {np.std(f1s):.4f}\n")




def compute_TSTR_CORAL(dataset):
    accs = []
    f1s = []

    for i in range(num_runs):

        accs_run = []
        f1s_run = []

        for src_class in config[dataset]['class_names']:
            if src_class != 'WAL':
                continue

            print(f"Source class: {src_class}\n")

            # Load Df data
            x_df, y_df, k_df = get_data(dataset, 'df', src_class)
            # x_df, _, y_df, _ = train_test_split(x_df, y_df, train_size=0.1, stratify=y_df, shuffle=True, random_state=seed)
            # print("Warning: Using only small fraction of Df data")
            print(f'x_df.shape: {x_df.shape} | np.unique(y_df): {np.unique(y_df)} | np.unique(k_df): {np.unique(k_df)}\n')

            for domain in range(config[dataset]['num_df_domains'], config[dataset]['num_df_domains'] + config[dataset]['num_dp_domains']):
                print(f"Domain: {domain}")

                # Load Dp data
                x_dp_dom, y_dp_dom, k_dp_dom = get_data(dataset, 'dp', src_class, domain)
                print(f'x_dp_dom.shape: {x_dp_dom.shape} | np.unique(y_dp_dom): {np.unique(y_dp_dom)} | np.unique(k_dp_dom): {np.unique(k_dp_dom)}\n')

                # Train on Df data and Dp data with CORAL and evaluate on Dp data
                print('Training on Df data and Dp data with CORAL...')
                acc, loss, f1 = train_and_test(x_df, y_df, x_dp_dom, y_dp_dom, dataset, num_epochs=num_epochs, patience=patience, train_coral=True)
                save_scores(src_class, domain, acc, loss, f1, 'CORAL', dataset)
                accs_run.append(acc)
                f1s_run.append(f1)
                print(f'Source class: {src_class} | Domain: {domain} | Loss: {loss:.4f} | Accuracy: {acc:.4f} | F1: {f1:.4f}\n')

        print(f"Mean accuracy for run {i}: {np.mean(accs_run):.4f} +- {np.std(accs_run):.4f}")
        print(f"Mean F1 for run {i}: {np.mean(f1s_run):.4f} +- {np.std(f1s_run):.4f}\n")
        accs.append(np.mean(accs_run))
        f1s.append(np.mean(f1s_run))

    print(f"Mean accuracy: {np.mean(accs):.4f} +- {np.std(accs):.4f}")
    print(f"Mean F1: {np.mean(f1s):.4f} +- {np.std(f1s):.4f}\n")




def compute_TSTR_CORAL_aug(dataset):
    accs = []
    f1s = []

    for i in range(num_runs):

        accs_run = []
        f1s_run = []

        for src_class in config[dataset]['class_names']:
            if src_class != 'WAL':
                continue

            print(f"Source class: {src_class}\n")

            # Load Df data
            x_df, y_df, k_df = get_data(dataset, 'df', src_class)
            # x_df, _, y_df, _ = train_test_split(x_df, y_df, train_size=0.1, stratify=y_df, shuffle=True, random_state=seed)
            # print("Warning: Using only small fraction of Df data")
            print(f'x_df.shape: {x_df.shape} | np.unique(y_df): {np.unique(y_df)} | np.unique(k_df): {np.unique(k_df)}\n')

            for domain in range(config[dataset]['num_df_domains'], config[dataset]['num_df_domains'] + config[dataset]['num_dp_domains']):
                print(f"Domain: {domain}")

                # Load Dp data
                x_dp_dom, y_dp_dom, k_dp_dom = get_data(dataset, 'dp', src_class, domain)
                print(f'x_dp_dom.shape: {x_dp_dom.shape} | np.unique(y_dp_dom): {np.unique(y_dp_dom)} | np.unique(k_dp_dom): {np.unique(k_dp_dom)}\n')

                # Train on Df data and Dp data with CORAL and evaluate on Dp data
                print('Training on Df data and Dp data with CORAL...')
                acc, loss, f1 = train_and_test(x_df, y_df, x_dp_dom, y_dp_dom, dataset, num_epochs=num_epochs, patience=patience, train_coral=True, augment=True)
                save_scores(src_class, domain, acc, loss, f1, 'CORAL_aug', dataset)
                accs_run.append(acc)
                f1s_run.append(f1)
                print(f'Source class: {src_class} | Domain: {domain} | Loss: {loss:.4f} | Accuracy: {acc:.4f} | F1: {f1:.4f}\n')

        print(f"Mean accuracy for run {i}: {np.mean(accs_run):.4f} +- {np.std(accs_run):.4f}")
        print(f"Mean F1 for run {i}: {np.mean(f1s_run):.4f} +- {np.std(f1s_run):.4f}\n")
        accs.append(np.mean(accs_run))
        f1s.append(np.mean(f1s_run))

    print(f"Mean accuracy: {np.mean(accs):.4f} +- {np.std(accs):.4f}")
    print(f"Mean F1: {np.mean(f1s):.4f} +- {np.std(f1s):.4f}\n")




def compute_TSTR_PL(dataset):
    accs = []
    f1s = []

    for i in range(num_runs):

        accs_run = []
        f1s_run = []

        for src_class in config[dataset]['class_names']:
            if src_class != 'WAL':
                continue

            print(f"Source class: {src_class}\n")

            # Load Df data
            x_df, y_df, k_df = get_data(dataset, 'df', src_class)
            x_df, _, y_df, _ = train_test_split(x_df, y_df, train_size=0.5, stratify=y_df, shuffle=True, random_state=seed)
            print("Warning: Using only small fraction of Df data")
            print(f'x_df.shape: {x_df.shape} | np.unique(y_df): {np.unique(y_df)} | np.unique(k_df): {np.unique(k_df)}\n')

            # Train on Df data
            print('Training on Df data...')
            df_model = train_only(x_df, y_df, dataset, num_epochs=num_epochs, patience=patience)

            for domain in range(config[dataset]['num_df_domains'], config[dataset]['num_df_domains'] + config[dataset]['num_dp_domains']):
                print(f"Domain: {domain}")

                # Load Dp data
                x_dp_dom, y_dp_dom, k_dp_dom = get_data(dataset, 'dp', src_class, domain)
                print(f'x_dp_dom.shape: {x_dp_dom.shape} | np.unique(y_dp_dom): {np.unique(y_dp_dom)} | np.unique(k_dp_dom): {np.unique(k_dp_dom)}')

                # Evaluate on Dp data with pseudo-labeling
                acc, loss, f1 = pseudo_labeling(df_model, x_dp_dom, y_dp_dom, dataset, num_epochs=num_epochs, patience=patience)
                save_scores(src_class, domain, acc, loss, f1, 'PL', dataset)
                accs_run.append(acc)
                f1s_run.append(f1)
                print(f'Source class: {src_class} | Domain: {domain} | Loss: {loss:.4f} | Accuracy: {acc:.4f} | F1: {f1:.4f}\n')

        print(f"Mean accuracy for run {i}: {np.mean(accs_run):.4f} +- {np.std(accs_run):.4f}")
        print(f"Mean F1 for run {i}: {np.mean(f1s_run):.4f} +- {np.std(f1s_run):.4f}\n")
        accs.append(np.mean(accs_run))
        f1s.append(np.mean(f1s_run))

    print(f"Mean accuracy over {num_runs} runs: {np.mean(accs):.4f} +- {np.std(accs):.4f}")
    print(f"Mean F1 over {num_runs} runs: {np.mean(f1s):.4f} +- {np.std(f1s):.4f}\n")



