import pickle
import os
import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset




def get_data(dataset, domains_set, domain=None):

    if domains_set == 'df' and domain is not None:
        raise ValueError("Domain filtering is not supported for Df data")

    dataset_name = dataset + f'_{domains_set}'

    try:
        with open(f'data/splits/{dataset_name}.pkl', 'rb') as f:
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



def get_dataloader(x, y, shuffle=False, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.tensor(x, dtype=torch.float32, device=device)
    y = torch.tensor(y, dtype=torch.long, device=device)

    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

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

    return R


def augment_batch(x_real):
    """Apply random rotation to the batch of real time series."""
    min_val, max_val = -19.61, 19.61
    x_real = x_real * (max_val - min_val) + min_val  # De-normalize
    R = random_rotation_matrix()
    x_real = torch.matmul(R, x_real)  # Apply rotation
    x_real = (x_real - min_val) / (max_val - min_val)  # Re-normalize
    return x_real



def save_cm(cm, name, dataset, results_dir='results_prova'):
    # Ensure the directory exists
    os.makedirs(results_dir, exist_ok=True)

    title = f'{dataset}_{name}'

    classes = ['WAL', 'RUN', 'CLD', 'CLU']
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    
    # Calculate accuracy per class
    accuracy_per_class = cm.diagonal() / cm.sum(axis=1)
    
    # Print accuracy per class
    for i, class_name in enumerate(classes):
        print(f'Accuracy for {class_name}: {accuracy_per_class[i]:.4f}')
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=df_cm.astype(int), fmt="d", cmap="Blues")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.savefig(f'{results_dir}/{title}.png')
    plt.show()



def save_scores(domain, loss, accuracy, f1, name, dataset, results_dir='results_prova', step=0):
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
            writer.writerow(['step', 'domain', 'loss', 'accuracy', 'f1'])
        # Write the data rows
        writer.writerow([step, domain, loss, accuracy, f1])