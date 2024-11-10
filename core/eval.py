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


def run_evaluation(step, G, args):
    print(f'Running evaluation at iteration {step}...')

    start_time = time.time()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    domain_classifier_te = DomainClassifier(args.num_channels, args.num_dp_domains, args.num_classes, args.num_timesteps)
    filename = f'pretrained_nets/domain_classifier_{args.dataset}_dp.ckpt'
    domain_classifier_te.load_state_dict(torch.load(filename, map_location=device))
    domain_classifier_te = domain_classifier_te.to(device)

    # siamese_net_te = SiameseNet(args.num_channels, args.num_classes, args.num_timesteps)
    # filename = f'pretrained_nets/siamese_net_{args.dataset}_dp.ckpt'
    # siamese_net_te.load_state_dict(torch.load(filename, map_location=device))
    # # siamese_net_te.load_state_dict(torch.load(filename, map_location=device, weights_only=False))
    # siamese_net_te = siamese_net_te.to(device)
    
    classes_dict = {clss: i for i, clss in enumerate(args.class_names)}
    
    for src_class in args.class_names:

        # Skip if the class is not WAL
        if src_class != 'WAL':
            continue

        src_idx = classes_dict[src_class]
        x_src, y_src, k_src = get_data(args.dataset, src_idx, args.num_df_domains)
        x_src = torch.tensor(x_src, dtype=torch.float32).to(device)

        N = len(x_src)
        
        trg_classes = [clss for clss in args.class_names if clss != src_class]

        syn_data = []
        syn_labels = []
        syn_doms = []

        for trg_class in trg_classes:

            trg_idx = classes_dict[trg_class]
            y_trg = torch.tensor([trg_idx] * N).to(device)
            y_trg_oh = label2onehot(y_trg, args.num_classes)

            with torch.no_grad():
                x_fake = G(x_src, y_trg_oh.to(device))

            calculate_domain_scores(domain_classifier_te, x_fake, y_trg, k_src, src_class, trg_class, step, args)
            # calculate_dist_scores(siamese_net_te, x_fake, y_trg, k_src, src_class, trg_class, step, args)
            
            syn_data.append(x_fake)
            syn_labels.append(y_trg)
            syn_doms.append(k_src)

        syn_data = torch.cat(syn_data, dim=0).cpu().detach().numpy()
        syn_labels = torch.cat(syn_labels, dim=0).cpu().detach().numpy()
        syn_doms = np.concatenate(syn_doms, axis=0)

        calculate_classification_scores(syn_data, syn_labels, syn_doms, src_class, trg_classes, step, args)

    print(f'Total time taken: {time.time() - start_time:.2f} seconds\n')



def get_data(dataset_name, class_idx, num_df_domains):

    # Load the dataset
    with open(f'data/{dataset_name}.pkl', 'rb') as f:
        x, y, k = pickle.load(f)

    with open(f'data/{dataset_name}_fs.pkl', 'rb') as f:
        fs = pickle.load(f)

    # Filter out the samples that are used for finetuning
    x = x[fs == 0]
    y = y[fs == 0]
    k = k[fs == 0]
    
    x_ = x[(y == class_idx) & (k >= num_df_domains)]
    y_ = y[(y == class_idx) & (k >= num_df_domains)]
    k_ = k[(y == class_idx) & (k >= num_df_domains)] - num_df_domains

    return x_, y_, k_



def label2onehot(labels, dim):
    """Convert label indices to one-hot vectors."""
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim)
    out[np.arange(batch_size), labels.long()] = 1
    return out


class DomainClassifier(nn.Module):
    def __init__(self, num_channels=3, num_domains=4, num_classes=5, num_timesteps=128):
        super(DomainClassifier, self).__init__()
        # Shared layers for all branches
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

        # Prepare class-specific branches as a single module with conditionally applied outputs
        self.fc_class_branches = nn.Linear(100, 50 * num_classes)
        self.fc_final = nn.Linear(50, num_domains)

    def forward(self, x, class_ids):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.relu(self.fc_shared(x))

        # Process all class-specific branches simultaneously
        class_branches = self.fc_class_branches(x).view(x.size(0), -1, 50)
        class_outputs = class_branches[torch.arange(class_branches.size(0)), class_ids]

        # Final class-specific output
        final_outputs = self.fc_final(class_outputs.view(x.size(0), 50))
        return final_outputs.view(x.size(0), -1)


class SiameseNet(nn.Module):
    def __init__(self, num_channels=3, num_classes=5, num_timesteps=128):
        super(SiameseNet, self).__init__()
        # Shared layers
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

        # Class-specific branches
        self.fc_class_branches = nn.Linear(100, 50 * num_classes)

    def forward_once(self, x, class_id):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.relu(self.fc_shared(x))

        # Process class-specific branch
        class_branches = self.fc_class_branches(x).view(x.size(0), -1, 50)
        class_output = class_branches[torch.arange(class_branches.size(0)), class_id]
        return class_output

    def forward(self, input1, input2, class_id1, class_id2):
        output1 = self.forward_once(input1, class_id1)
        output2 = self.forward_once(input2, class_id2)
        return output1, output2
    

class TSTRClassifier(nn.Module):
    def __init__(self, num_timesteps=128, num_channels=3, num_classes=5):
        super(TSTRClassifier, self).__init__()

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

        self.fc_class = nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)
        x = self.relu(self.fc_shared(x))
        
        # Final output for class prediction
        class_outputs = self.fc_class(x)
        return class_outputs



def calculate_domain_scores(domain_classifier, x_fake, y_trg, k_fake, src_class, trg_class, step, args):
    print(f'Calculating domain score for {src_class} -> {trg_class}...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    out = domain_classifier(x_fake, y_trg)
    
    loss = nn.CrossEntropyLoss()
    loss_val = loss(out, torch.tensor(k_fake, dtype=torch.long).to(device)).item()

    preds = torch.argmax(out, dim=1).detach().cpu().numpy()
    accuracy = np.mean(preds == k_fake)

    print(f'Accuracy: {accuracy:.4f}, Loss: {loss_val:.4f}\n')

    save_domain_scores((accuracy, loss_val), src_class, trg_class, step, args.results_dir)



def calculate_dist_scores(siamese_net_te, x_fake, y_trg, k_fake, src_class, trg_class, step, args):
    print(f'Calculating distance score for {src_class} -> {trg_class}...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    classes_dict = {clss: i for i, clss in enumerate(args.class_names)}
    trg_idx = classes_dict[trg_class]

    x_real, y_real, k_real = get_data(args.dataset, trg_idx, args.num_df_domains)
    
    # Ensure k_real and k_fake contain the same set of unique elements
    assert np.array_equal(np.unique(k_real), np.unique(k_fake)), 'k_real and k_fake contain different unique elements'

    siamese_net_te.eval()
    with torch.no_grad():
        real_features = siamese_net_te.forward_once(torch.tensor(x_real, dtype=torch.float32).to(device), 
                                                    torch.tensor(y_real, dtype=torch.long).to(device))
        fake_features = siamese_net_te.forward_once(x_fake.clone().detach().to(device).float(),
                                                    y_trg.clone().detach().to(device).long())
    
    # Calculate average distance for each unique k
    unique_k = np.unique(k_real)
    avg_distances = {}
    for k in unique_k:
        real_indices = [i for i, val in enumerate(k_real) if val == k]
        fake_indices = [i for i, val in enumerate(k_fake) if val == k]
        
        real_k_features = real_features[real_indices]
        fake_k_features = fake_features[fake_indices]

        real_k_features_exp = real_k_features.unsqueeze(1)
        fake_k_features_exp = fake_k_features.unsqueeze(0)
        
        distances = F.pairwise_distance(real_k_features_exp, fake_k_features_exp, keepdim=True)
        avg_distance = distances.mean().item()
        avg_distances[k] = avg_distance

    avg_avg_distance = np.mean(list(avg_distances.values()))
    print(f'Average distance: {avg_avg_distance:.4f}\n')

    save_dist_scores(avg_distances, src_class, trg_class, step, args.results_dir)



def calculate_classification_scores(syn_data, syn_labels, syn_doms, src_class, trg_classes, step, args):

    print('Calculating TSTR score for %s source...\n' % src_class)

    classes_dict = {clss: i for i, clss in enumerate(args.class_names)}

    trg_data = []
    trg_labels = []
    trg_doms = []   

    for trg_class in trg_classes:

        trg_idx = classes_dict[trg_class]
        x_trg, y_trg, k_trg = get_data(args.dataset, trg_idx, args.num_df_domains)

        trg_data.append(x_trg)
        trg_labels.append(y_trg)
        trg_doms.append(k_trg)

    trg_data = np.concatenate(trg_data, axis=0)
    trg_labels = np.concatenate(trg_labels, axis=0)
    trg_doms = np.concatenate(trg_doms, axis=0)

    assert np.array_equal(np.unique(syn_doms), np.unique(trg_doms))
    assert np.array_equal(np.unique(syn_labels), np.unique(trg_labels))

    accs = []
    loglosses = []

    for domain in np.unique(syn_doms):
        syn_data_dom = syn_data[syn_doms == domain]
        trg_data_dom = trg_data[trg_doms == domain]

        syn_labels_dom = syn_labels[syn_doms == domain]
        trg_labels_dom = trg_labels[trg_doms == domain]

        print(f'\nSource: {src_class}, Domain: {domain+args.num_df_domains}, Target: {trg_classes}, Syn data: {syn_data_dom.shape}, Trg data: {trg_data_dom.shape}')
        
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(np.unique(syn_labels))}
        syn_labels_dom = np.array([label_mapping[x] for x in syn_labels_dom])
        trg_labels_dom = np.array([label_mapping[x] for x in trg_labels_dom])

        acc, logloss = compute_accuracy(syn_data_dom, syn_labels_dom, trg_data_dom, trg_labels_dom)

        print(f'Source: {src_class}, Domain: {domain}, Accuracy: {acc:.4f}, Logloss: {logloss:.4f}')
        classification_scores = (acc, logloss)
        save_classification_scores(classification_scores, src_class, domain, step, args.results_dir, args.num_df_domains)

        accs.append(acc)
        loglosses.append(logloss)

    print(f'\nMean accuracy: {np.mean(accs):.4f}, Mean logloss: {np.mean(loglosses):.4f}\n\n')

    return accs, loglosses


def compute_accuracy(x_train, y_train, x_test, y_test):
    x_tr, x_val, y_tr, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=2710, stratify=y_train, shuffle=True)
    tr_loader, val_loader, test_loader = setup_training(x_tr, y_tr, x_val, y_val, x_test, y_test, batch_size=64)
    
    model = TSTRClassifier(num_timesteps=x_train.shape[2], num_channels=x_train.shape[1], num_classes=len(np.unique(y_train)))
    loss_fn = nn.CrossEntropyLoss()
    initial_lr = 0.0001
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)

    best_model_state = train_model(model, tr_loader, val_loader, loss_fn, optimizer, epochs=100)
    best_model = TSTRClassifier(num_timesteps=x_train.shape[2], num_channels=x_train.shape[1], num_classes=len(np.unique(y_train)))
    best_model.load_state_dict(best_model_state)
    test_accuracy, test_loss = evaluate_model(best_model, test_loader, loss_fn)

    return test_accuracy, test_loss



def setup_training(x_tr, y_tr, x_val, y_val, x_test, y_test, batch_size=64):
    # Convert numpy arrays to torch tensors
    x_train_tensor = torch.tensor(x_tr, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_tr, dtype=torch.long)
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Create datasets and loaders
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train_model(model, train_loader, val_loader, loss_fn, optimizer, epochs=300):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loss_train = []
    loss_val = []
    accuracy_val = []
    best_loss = np.inf
    best_accuracy = 0

    # Set up linear learning rate decay
    lambda_lr = lambda epoch: 1 - epoch / epochs
    scheduler = LambdaLR(optimizer, lr_lambda=lambda_lr)

    for epoch in range(epochs):
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
        loss_train.append(total_loss)

        # Update learning rate
        scheduler.step()

        val_accuracy, val_loss = evaluate_model(model, val_loader, loss_fn)
        if val_accuracy > best_accuracy:
            best_epoch = epoch
            best_accuracy = val_accuracy
            best_loss = val_loss
            best_model_state = model.state_dict().copy()
        loss_val.append(val_loss)
        accuracy_val.append(val_accuracy)

        current_lr = scheduler.get_last_lr()[0]
        if (epoch+1) % 20 == 0:
            print(f"\tEpoch {epoch + 1}/{epochs} - Train loss: {total_loss:.4f} - Val loss: {val_loss:.4f} - Val accuracy: {val_accuracy:.4f} - LR: {current_lr:.6f}")
    
    print(f"\tBest epoch: {best_epoch + 1} - Best val accuracy: {best_accuracy:.4f} - Best val loss: {best_loss:.4f}")

    return best_model_state


def evaluate_model(model, test_loader, loss_fn):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            loss = loss_fn(outputs, y_batch)
            total_loss += loss.item()

            _, predicted_labels = torch.max(outputs, 1)
            correct_predictions += (predicted_labels == y_batch).sum().item()
            total_predictions += len(y_batch)

    total_loss /= len(test_loader)
    accuracy = correct_predictions / total_predictions

    return accuracy, total_loss


def save_domain_scores(domain_scores, src_class, trg_class, step, results_dir):
    # Ensure the directory exists
    os.makedirs(results_dir, exist_ok=True)
    # Path to the CSV file
    file_path = os.path.join(results_dir, 'domain_scores.csv')
    # Check if the file exists
    file_exists = os.path.exists(file_path)
    
    # Open the file in append mode if it exists, or write mode if it doesn't
    with open(file_path, mode='a' if file_exists else 'w', newline='') as file:
        writer = csv.writer(file)
        # If the file does not exist, write the header
        if not file_exists:
            writer.writerow(['step', 'source', 'target', 'accuracy', 'loss'])
        
        accuracy, loss = domain_scores
        # Write the data rows
        writer.writerow([step, src_class, trg_class, accuracy, loss])



def save_dist_scores(dist_scores, src_class, trg_class, step, results_dir):
    # Ensure the directory exists
    os.makedirs(results_dir, exist_ok=True)
    # Path to the CSV file
    file_path = os.path.join(results_dir, 'dist_scores.csv')
    # Check if the file exists
    file_exists = os.path.exists(file_path)
    
    # Open the file in append mode if it exists, or write mode if it doesn't
    with open(file_path, mode='a' if file_exists else 'w', newline='') as file:
        writer = csv.writer(file)
        # If the file does not exist, write the header
        if not file_exists:
            writer.writerow(['step', 'source', 'target', 'domain', 'distance'])
            
        # Write the data rows
        for k, distance in dist_scores.items():
            writer.writerow([step, src_class, trg_class, k, distance])


def save_classification_scores(classification_scores, src_class, domain, step, results_dir, num_df_domains):
    # Ensure the directory exists
    os.makedirs(results_dir, exist_ok=True)
    # Path to the CSV file
    file_path = os.path.join(results_dir, 'TSTR_scores.csv')
    # Check if the file exists
    file_exists = os.path.exists(file_path)
    
    # Open the file in append mode if it exists, or write mode if it doesn't
    with open(file_path, mode='a' if file_exists else 'w', newline='') as file:
        writer = csv.writer(file)
        # If the file does not exist, write the header
        if not file_exists:
            writer.writerow(['step', 'source', 'domain', 'accuracy', 'loss'])

        accuracy, loss = classification_scores
        # Write the data rows
        writer.writerow([step, src_class, domain+num_df_domains, accuracy, loss])
