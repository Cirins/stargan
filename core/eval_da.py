import torch
import numpy as np
import os
import csv
import time
import pickle
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from core.TSTR_scores_da.train_functions import train_and_test
from core.TSTR_scores_da.utils import save_cm, save_scores


def run_evaluation(step, G, args):
    print(f'\nRunning evaluation at iteration {step}...\n')

    start_time = time.time()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x_syn, y_syn, k_syn, x_dp_te, y_dp_te, k_dp_te = get_data(G, args, device)

    accs, f1s = [], []
    total_cm = None

    for domain in range(args.num_df_domains, args.num_df_domains + args.num_dp_domains):
        print(f"Domain: {domain}")

        # Filter only the domain samples
        mask_syn = k_syn == domain
        x_syn_dom, y_syn_dom, k_syn_dom = x_syn[mask_syn], y_syn[mask_syn], k_syn[mask_syn]
        print(f'x_syn_dom.shape: {x_syn_dom.shape} | np.unique(y_syn_dom): {np.unique(y_syn_dom)} | np.unique(k_syn_dom): {np.unique(k_syn_dom)}')

        mask_dp = k_dp_te == domain
        x_dp_dom, y_dp_dom, k_dp_dom = x_dp_te[mask_dp], y_dp_te[mask_dp], k_dp_te[mask_dp]
        print(f'x_dp_dom.shape: {x_dp_dom.shape} | np.unique(y_dp_dom): {np.unique(y_dp_dom)} | np.unique(k_dp_dom): {np.unique(k_dp_dom)}')

        # Train on synthetic data and evaluate on Dp data
        print('Training on synthetic data...')
        loss, acc, f1, cm = train_and_test(x_syn_dom, y_syn_dom, x_dp_dom, y_dp_dom, num_epochs=40)
        save_scores(domain, loss, acc, f1, 'TSTR', args.dataset, args.results_dir, step)
        accs.append(acc)
        f1s.append(f1)
        print(f'Domain: {domain} | Loss: {loss:.4f} | Accuracy: {acc:.4f} | F1: {f1:.4f}\n')
        if total_cm is None:
            total_cm = cm
        else:
            total_cm += cm

    print(f"Mean accuracy: {np.mean(accs):.4f} +- {np.std(accs):.4f}")
    print(f"Mean F1: {np.mean(f1s):.4f} +- {np.std(f1s):.4f}\n")
    save_cm(total_cm, step, args.dataset, args.results_dir)

    print(f'Total time taken: {time.time() - start_time:.2f} seconds\n')



def get_data(G, args, device):

    # Load the dataset
    with open(f'data/splits/{args.dataset}_dp_map.pkl', 'rb') as f:
        x_dp_map, y_dp_map, k_dp_map = pickle.load(f)
    print(f'Loaded Dp map data with shape {x_dp_map.shape}, from {len(set(k_dp_map))} domains and {len(set(y_dp_map))} classes')

    # Create tensors
    x_dp_map = torch.tensor(x_dp_map, dtype=torch.float32, device=device)
    k_dp_map = torch.tensor(k_dp_map, dtype=torch.long, device=device)
    y_dp_map = torch.tensor(y_dp_map, dtype=torch.long, device=device)

    # Map x to the target classes
    x_syn, y_syn, k_syn = [], [], []
    with torch.no_grad():
        for y_trg in range(0, args.num_classes):
            print(f'Mapping to class {y_trg}...')
            y_trg_tensor = torch.tensor([y_trg] * x_dp_map.size(0), device=device)
            y_trg_oh = label2onehot(y_trg_tensor, args.num_classes)
            x_syn.append(G(x_dp_map, y_trg_oh))
            y_syn.append(y_trg_tensor)
            k_syn.append(k_dp_map)
        
    x_syn = torch.cat(x_syn, dim=0).detach().cpu().numpy()
    y_syn = torch.cat(y_syn, dim=0).detach().cpu().numpy()
    k_syn = torch.cat(k_syn, dim=0).detach().cpu().numpy()
    print(f'Loaded Syn data with shape {x_syn.shape}, from {len(set(k_syn))} domains and {len(set(y_syn))} classes')

    # Load the dataset
    with open(f'data/splits/{args.dataset}_dp_te.pkl', 'rb') as f:
        x_dp_te, y_dp_te, k_dp_te = pickle.load(f)
    print(f'Loaded Dp test data with shape {x_dp_te.shape}, from {len(set(k_dp_te))} domains and {len(set(y_dp_te))} classes\n')

    return x_syn, y_syn, k_syn, x_dp_te, y_dp_te, k_dp_te



def label2onehot(labels, dim):
    """Convert label indices to one-hot vectors."""
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim, device=labels.device)
    out[np.arange(batch_size), labels.long()] = 1
    return out



