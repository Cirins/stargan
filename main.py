import os
import argparse
from core.solver import Solver
from core.data_loader import get_dataloaders
import torch
import numpy as np
from torch.backends import cudnn
import random


def str2bool(v):
    return v.lower() in ('true')

def main(args):
    # For fast training.
    cudnn.benchmark = True

    # Set random seed.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Create directories if not exist.
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_save_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # Data loaders.
    train_loader, test_loader = get_dataloaders(args.dataset, args.class_names, args.num_df_domains, args.batch_size, args.num_workers, args.finetune)    

    # Solver for training and testing StarGAN.
    solver = Solver(train_loader, test_loader, args)

    if args.mode == 'train':
        solver.train()
    elif args.mode == 'finetune':
        solver.train()
    elif args.mode == 'sample':
        solver.sample(args.syn_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--num_timesteps', type=int, default=128, help='number of timesteps in the input time series')
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    parser.add_argument('--lambda_dom', type=float, default=1, help='weight for domain loss')
    parser.add_argument('--lambda_rot', type=float, default=10, help='weight for rotation loss')
    parser.add_argument('--loss_type', type=str, default='lsgan', choices=['gan', 'lsgan', 'wgan-gp'], help='type of GAN loss')
    
    # Training configuration.
    parser.add_argument('--dataset', type=str, default='realworld_mobiact', choices=['realworld', 'cwru', 'realworld_mobiact'], help='dataset name')
    parser.add_argument('--class_names', type=str, required=True, help='class names')
    parser.add_argument('--channel_names', type=str, required=True, help='channel names')
    parser.add_argument('--num_df_domains', type=int, required=True, help='number of domains in Df')
    parser.add_argument('--num_dp_domains', type=int, required=True, help='number of domains in Dp')
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=1000000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    parser.add_argument('--augment', type=str2bool, default=False, help='augment data')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'finetune', 'sample'])
    parser.add_argument('--seed', type=int, default=2710, help='random seed for training')
    parser.add_argument('--syn_name', type=str, default='syn', help='name of the synthetic dataset')

    # Step size.
    parser.add_argument('--log_step', type=int, default=100)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=10000)
    parser.add_argument('--eval_step', type=int, default=10000)
    parser.add_argument('--lr_update_step', type=int, default=-1, help='lr update step, set -1 to disable')

    args = parser.parse_args()

    args.class_names = args.class_names.split()
    args.num_classes = len(args.class_names)
    args.channel_names = args.channel_names.split()
    args.num_channels = len(args.channel_names)

    if args.mode == 'finetune':
        args.finetune = True
    else:
        args.finetune = False

    args.expr_dir = f'expr_{args.dataset}' 
    args.log_dir = os.path.join(args.expr_dir, 'logs')
    args.model_save_dir = os.path.join(args.expr_dir, 'models')
    args.sample_dir = os.path.join(args.expr_dir, 'samples')
    args.results_dir = os.path.join(args.expr_dir, 'results')

    print(args)
    main(args)