import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import LambdaLR
import torch.optim as optim
import os
import csv
import time
import pickle
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def run_evaluation(step, G, args):
    print(f'Running evaluation at iteration {step}...\n')

    start_time = time.time()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x_syn, y_syn, k_syn, x_dp, y_dp, k_dp = get_data(G, args, device)

    accs, f1s = [], []
    total_cm = None

    for domain in range(args.num_df_domains, args.num_df_domains + args.num_dp_domains):
        print(f"Domain: {domain}")

        # Filter only the domain samples
        mask_syn = k_syn == domain
        x_syn_dom, y_syn_dom, k_syn_dom = x_syn[mask_syn], y_syn[mask_syn], k_syn[mask_syn]
        print(f'x_syn_dom.shape: {x_syn_dom.shape} | np.unique(y_syn_dom): {np.unique(y_syn_dom)} | np.unique(k_syn_dom): {np.unique(k_syn_dom)}')

        mask_dp = k_dp == domain
        x_dp_dom, y_dp_dom, k_dp_dom = x_dp[mask_dp], y_dp[mask_dp], k_dp[mask_dp]
        print(f'x_dp_dom.shape: {x_dp_dom.shape} | np.unique(y_dp_dom): {np.unique(y_dp_dom)} | np.unique(k_dp_dom): {np.unique(k_dp_dom)}')

        # Train on synthetic data and evaluate on Dp data
        print('Training on synthetic data...')
        acc, loss, f1, cm = train_and_test(x_syn_dom, y_syn_dom, x_dp_dom, y_dp_dom, args, num_epochs=40)
        save_scores(step, domain, acc, loss, f1, 'Syn', args)
        accs.append(acc)
        f1s.append(f1)
        print(f'Domain: {domain} | Loss: {loss:.4f} | Accuracy: {acc:.4f} | F1: {f1:.4f}\n')

        if total_cm is None:
            total_cm = cm
        else:
            total_cm += cm

    print(f"Mean accuracy: {np.mean(accs):.4f} +- {np.std(accs):.4f}")
    print(f"Mean F1: {np.mean(f1s):.4f} +- {np.std(f1s):.4f}\n")
    save_cm(total_cm, step, 'Syn', args)



def get_data(G, args, device):

    # Load the dataset
    with open(f'data/{args.dataset}.pkl', 'rb') as f:
        x, y, k = pickle.load(f)

    print(f'Loaded full dataset with shape {x.shape}, from {len(set(k))} domains and {len(set(y))} classes')
    
    # Filter only df domains
    mask_df = (k < args.num_df_domains)
    x_df = x[mask_df]
    k_df = k[mask_df]
    y_df = y[mask_df]

    print(f'Loaded Df data with shape {x_df.shape}, from {len(set(k_df))} domains and {len(set(y_df))} classes')
    
    # Filter only class 0 samples and dp domains
    mask_dp_0 = (y == 0) & (k >= args.num_df_domains)
    x_dp_0 = x[mask_dp_0]
    k_dp_0 = k[mask_dp_0]
    y_dp_0 = y[mask_dp_0]

    print(f'Loaded class 0 Dp data with shape {x_dp_0.shape}, from {len(set(k_dp_0))} domains and {len(set(y_dp_0))} classes')

    x_dp_0_map, x_dp_0_te, k_dp_0_map, k_dp_0_te, y_dp_0_map, y_dp_0_te = train_test_split(x_dp_0, k_dp_0, y_dp_0, test_size=0.2, random_state=2710, stratify=k_dp_0, shuffle=True)

    print(f'Divided class 0 Dp data into map with shape {x_dp_0_map.shape}, from {len(set(k_dp_0_map))} domains and {len(set(y_dp_0_map))} classes')
    print(f'And into test with shape {x_dp_0_te.shape}, from {len(set(k_dp_0_te))} domains and {len(set(y_dp_0_te))} classes')

    # Create tensors
    x_dp_0_map = torch.tensor(x_dp_0_map, dtype=torch.float32, device=device)
    k_dp_0_map = torch.tensor(k_dp_0_map, dtype=torch.long, device=device)
    y_dp_0_map = torch.tensor(y_dp_0_map, dtype=torch.long, device=device)

    x_syn, y_syn, k_syn = [x_dp_0_map], [y_dp_0_map], [k_dp_0_map]

    # Map x to the target classes
    with torch.no_grad():
        for y_trg in range(1, args.num_classes):
            print(f'Mapping class 0 to class {y_trg}...')
            y_trg_tensor = torch.tensor([y_trg] * x_dp_0_map.size(0), device=device)
            y_trg_oh = label2onehot(y_trg_tensor, args.num_classes)
            x_syn.append(G(x_dp_0_map, y_trg_oh))
            y_syn.append(y_trg_tensor)
            k_syn.append(k_dp_0_map)
        
    x_syn = torch.cat(x_syn, dim=0).detach().cpu().numpy()
    y_syn = torch.cat(y_syn, dim=0).detach().cpu().numpy()
    k_syn = torch.cat(k_syn, dim=0).detach().cpu().numpy()

    print(f'Loaded Syn data with shape {x_syn.shape}, from {len(set(k_syn))} domains and {len(set(y_syn))} classes')
    
    # Filter not class 0 samples and dp domains
    mask_dp_not0 = (y != 0) & (k >= args.num_df_domains)
    x_dp_not0 = x[mask_dp_not0]
    k_dp_not0 = k[mask_dp_not0]
    y_dp_not0 = y[mask_dp_not0]

    print(f'Loaded classes not0 Dp data with shape {x_dp_not0.shape}, from {len(set(k_dp_not0))} domains and {len(set(y_dp_not0))} classes')

    x_dp = np.concatenate([x_dp_0_te, x_dp_not0], axis=0)
    y_dp = np.concatenate([y_dp_0_te, y_dp_not0], axis=0)
    k_dp = np.concatenate([k_dp_0_te, k_dp_not0], axis=0)

    print(f'Loaded Dp data with shape {x_dp.shape}, from {len(set(k_dp))} domains and {len(set(y_dp))} classes\n')

    return x_syn, y_syn, k_syn, x_dp, y_dp, k_dp



def label2onehot(labels, dim):
    """Convert label indices to one-hot vectors."""
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim, device=labels.device)
    out[np.arange(batch_size), labels.long()] = 1
    return out



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
        self.fc_shared = nn.Linear(num_timesteps * 8, 100)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)
        x = self.relu(self.fc_shared(x))
        return x


class TSTRClassifier(nn.Module):
    def __init__(self, num_timesteps=128, num_channels=3, num_classes=5):
        super(TSTRClassifier, self).__init__()

        self.feature_extractor = TSTRFeatureExtractor(num_timesteps, num_channels)
        self.fc_class = nn.Linear(100, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.feature_extractor(x)
        class_outputs = self.fc_class(x)
        return class_outputs
    


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
    cm = confusion_matrix(all_labels, all_preds)
    total_loss /= len(test_loader)

    return accuracy, total_loss, f1, cm



def train_model(model, train_loader, val_loader, optimizer, num_epochs=40):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        total_loss /= len(train_loader)

        if (epoch+1) % 10 == 0:
            print(f"\tEpoch {epoch + 1}/{num_epochs} - Train loss: {total_loss:.4f}")

    return model



def train_and_test(x_train, y_train, x_test, y_test, args, num_epochs=40):
    assert np.array_equal(np.unique(y_train), np.unique(y_test)), f"Training and test labels do not match: {np.unique(y_train)} vs {np.unique(y_test)}"

    num_classes = len(np.unique(y_train))

    train_loader = get_dataloader(x_train, y_train, shuffle=True)
    val_loader = None
    test_loader = get_dataloader(x_test, y_test)

    model = TSTRClassifier(num_timesteps=args.num_timesteps,
                           num_channels=args.num_channels,
                           num_classes=num_classes)
    initial_lr = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=1e-4)

    trained_model = train_model(model, train_loader, val_loader, optimizer, num_epochs=num_epochs)

    test_accuracy, test_loss, test_f1, test_cm = evaluate_model(trained_model, test_loader)

    return test_accuracy, test_loss, test_f1, test_cm



def save_scores(step, domain, accuracy, loss, f1, name, args):
    # Ensure the directory exists
    os.makedirs(args.results_dir, exist_ok=True)
    # Path to the CSV file
    file_path = os.path.join(args.results_dir, f'{args.dataset}_{name}.csv')
    # Check if the file exists
    file_exists = os.path.exists(file_path)

    # Open the file in append mode if it exists, or write mode if it doesn't
    with open(file_path, mode='a' if file_exists else 'w', newline='') as file:
        writer = csv.writer(file)
        # If the file does not exist, write the header
        if not file_exists:
            writer.writerow(['step', 'domain', 'accuracy', 'f1', 'loss'])
        # Write the data rows
        writer.writerow([step, domain, accuracy, f1, loss])


def save_cm(cm, step, name, args):
    # Ensure the directory exists
    os.makedirs(args.results_dir, exist_ok=True)

    classes = ['WAL', 'RUN', 'CLD', 'CLU']
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    
    # Calculate accuracy per class
    accuracy_per_class = cm.diagonal() / cm.sum(axis=1)
    
    # Print accuracy per class
    for i, class_name in enumerate(classes):
        print(f'Accuracy for {class_name}: {accuracy_per_class[i]:.4f}')
    print()
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=df_cm.astype(int), fmt="d", cmap="Blues")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{args.dataset} - {name}')
    plt.savefig(f'{args.results_dir}/{args.dataset}_{name}_cm_{step}.png')
    plt.show()



