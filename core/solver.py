from core.model import Generator, Discriminator
from core.eval_da import run_evaluation
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
import csv
import matplotlib.pyplot as plt
import random
import pickle
from sklearn.model_selection import train_test_split


class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, train_loader, test_loader, td_loader, args):
        """Initialize configurations."""

        self.args = args

        # Data loader.
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.td_loader = td_loader

        # Model configurations.
        self.num_timesteps = args.num_timesteps
        self.g_conv_dim = args.g_conv_dim
        self.d_conv_dim = args.d_conv_dim
        self.g_repeat_num = args.g_repeat_num
        self.d_repeat_num = args.d_repeat_num
        self.lambda_cls = args.lambda_cls
        self.lambda_rec = args.lambda_rec
        self.lambda_gp = args.lambda_gp
        self.lambda_dom = args.lambda_dom
        self.lambda_rot = args.lambda_rot
        self.loss_type = args.loss_type

        # Training configurations.
        self.dataset = args.dataset
        self.class_names = args.class_names
        self.num_classes = args.num_classes
        self.channel_names = args.channel_names
        self.num_channels = args.num_channels
        self.num_df_domains = args.num_df_domains
        self.num_dp_domains = args.num_dp_domains
        self.batch_size = args.batch_size
        self.num_iters = args.num_iters
        self.num_iters_decay = args.num_iters_decay
        self.lr_update_step = args.lr_update_step
        self.g_lr = args.g_lr
        self.d_lr = args.d_lr
        self.n_critic = args.n_critic
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.resume_iters = args.resume_iters
        self.augment = args.augment
        self.min_g_lr = args.min_g_lr
        self.min_d_lr = args.min_d_lr

        # Learning rate decay.
        self.total_decay_steps = (self.num_iters - self.num_iters_decay) // self.lr_update_step
        self.g_lr_step = (self.g_lr - self.min_g_lr) / self.total_decay_steps
        self.d_lr_step = (self.d_lr - self.min_d_lr) / self.total_decay_steps

        # Miscellaneous.
        self.mode = args.mode
        self.finetune = args.finetune
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.expr_dir = args.expr_dir
        self.log_dir = args.log_dir
        self.sample_dir = args.sample_dir
        self.model_save_dir = args.model_save_dir
        self.results_dir = args.results_dir

        # Step size.
        self.log_step = args.log_step
        self.sample_step = args.sample_step
        self.model_save_step = args.model_save_step
        self.eval_step = args.eval_step

        # Initialize CSV file for logging
        self.initialize_csv_log()

        # Build the model.
        self.build_model()

    def initialize_csv_log(self):
        """Initialize CSV file for logging."""
        self.log_file = os.path.join(self.log_dir, 'log.csv')
        file_exists = os.path.isfile(self.log_file)
        with open(self.log_file, 'a', newline='') as csvfile:
            d_keys = ['loss_real', 'loss_fake', 'loss_cls', 'loss_dom', 'loss_gp', 'loss_td_real', 'loss_td_fake']
            g_keys = ['loss_fake', 'loss_rec', 'loss_cls', 'loss_dom', 'loss_rot', 'loss_td_fake']
            fieldnames = ['Elapsed Time', 'Iteration'] + [f'D/{key}' for key in d_keys] + [f'G/{key}' for key in g_keys]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()

    def build_model(self):
        """Create a generator and a discriminator."""
        self.G = Generator(self.g_conv_dim, self.num_classes, self.g_repeat_num)
        num_domains = self.num_dp_domains if self.finetune else self.num_df_domains
        self.D = Discriminator(self.num_timesteps, self.d_conv_dim, self.num_classes, num_domains, self.d_repeat_num)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.print_network(self.G, 'Generator')
        self.print_network(self.D, 'Discriminator')
            
        self.G.to(self.device)
        self.D.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        # print(model)
        print(f'The number of parameters of the {name} is {num_params}')

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        
        if self.finetune:
            raise NotImplementedError('Restoring conv dom not implemented yet')
            # Load discriminator weights except for conv_dom
            D_state_dict = torch.load(D_path, map_location=lambda storage, loc: storage)
            D_state_dict = {k: v for k, v in D_state_dict.items() if not k.startswith('layers.conv_dom')}
            self.D.load_state_dict(D_state_dict, strict=False)
            
            # Reinitialize conv_dom layer
            self.D.reinitialize_last_layer()

            # Make only conv_src, conv_cls, and conv_dom trainable
            for name, param in self.D.named_parameters():
                if not (name.startswith('layers.conv_src') or name.startswith('layers.conv_cls') or name.startswith('layers.conv_dom') or name.startswith('layers.conv_rot')):
                    param.requires_grad = False

            # Make only upsampling layers and the last conv layer of G trainable
            for name, param in self.G.named_parameters():
                if not (name.startswith('layers.upsample') or name.startswith('layers.final')):
                    param.requires_grad = False

            # Print the trainable layers of G and D
            print('Trainable layers of G:')
            for name, param in self.G.named_parameters():
                if param.requires_grad:
                    print(name)
            print('Trainable layers of D:')
            for name, param in self.D.named_parameters():
                if param.requires_grad:
                    print(name)
            
            # Move the models to the device
            self.D.to(self.device)
            self.G.to(self.device)

            # Update optimizers to include only trainable parameters
            self.g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), self.g_lr, [self.beta1, self.beta2])
            self.d_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D.parameters()), self.d_lr, [self.beta1, self.beta2])
        else:
            self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def get_last_elapsed_time(self):
        """Retrieve the last elapsed time from the CSV log file."""
        if not os.path.isfile(self.log_file):
            return 0

        with open(self.log_file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
            if not rows:
                return 0

            last_row = rows[-1]
            elapsed_time_str = last_row['Elapsed Time']
            h, m, s = map(int, elapsed_time_str.split(':'))
            return h * 3600 + m * 60 + s

    def log_training_info(self, start_time, i, loss, initial_elapsed_time):
        """Log training information and save to CSV file."""
        et = time.time() - start_time + initial_elapsed_time
        et = str(datetime.timedelta(seconds=et))[:-7]
        log = {"Elapsed Time": et, "Iteration": i+1}
        log.update({key: f"{value:.4f}" for key, value in loss.items()})

        # Create a formatted string for the log dictionary
        log_str = f"Elapsed Time: {log['Elapsed Time']}, Iteration: {log['Iteration']}"
        for key, value in log.items():
            if key not in ['Elapsed Time', 'Iteration']:
                log_str += f", {key}: {value}"
        print(log_str)

        # Save log to CSV file
        with open(self.log_file, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=log.keys())
            writer.writerow(log)

    def save_time_series(self, data, labels, domains, filename):
        N = data.size(0)
        nrows = (self.num_classes + 1)
        ncols = N // nrows
        fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 2.5))
        axs = axs.flatten()
        for idx in range(N):
            for i in range(data.size(1)):
                axs[idx].plot(data[idx, i, :].detach().cpu().numpy(), label=self.channel_names[i], linewidth=0.7)
            axs[idx].set_ylim(0, 1)
            axs[idx].axis('off')
            axs[idx].set_title(f'{self.class_names[labels[idx].item()]} (Domain {domains[idx].item()})')
            if idx < ncols:
                axs[idx].legend(loc='lower left')
        for idx in range(N, len(axs)):
            axs[idx].axis('off')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    @torch.no_grad()
    def sample_time_series(self, x_fix, y_fix, k_fix, step, prefix):
        """Sample and save time series."""
        x_fix_list_map = [x_fix]
        x_fix_list_rec = [x_fix]
        y_fix_list_map = [y_fix]
        y_fix_list_rec = [y_fix]
        k_fix_list_map = [k_fix]
        k_fix_list_rec = [k_fix]
        for y in range(self.num_classes):
            # Map original class to target class.
            y_trg = torch.tensor([y] * x_fix.size(0)).to(self.device)
            y_trg_oh = self.label2onehot(y_trg, self.num_classes)
            x_fix_list_map.append(self.G(x_fix, y_trg_oh.to(self.device)))
            y_fix_list_map.append(y_trg)
            k_fix_list_map.append(k_fix)
            # Map target class to original class.
            y_src_oh = self.label2onehot(y_fix, self.num_classes)
            x_fix_list_rec.append(self.G(x_fix, y_src_oh.to(self.device)))
            y_fix_list_rec.append(y_fix)
            k_fix_list_rec.append(k_fix)
        x_fix_concat_map = torch.cat(x_fix_list_map, dim=0)
        y_fix_concat_map = torch.cat(y_fix_list_map, dim=0)
        k_fix_concat_map = torch.cat(k_fix_list_map, dim=0)
        x_fix_concat_rec = torch.cat(x_fix_list_rec, dim=0)
        y_fix_concat_rec = torch.cat(y_fix_list_rec, dim=0)
        k_fix_concat_rec = torch.cat(k_fix_list_rec, dim=0)
        self.save_time_series(x_fix_concat_map, y_fix_concat_map, k_fix_concat_map, os.path.join(self.sample_dir, f'{step:06d}_{prefix}_map.png'))
        self.save_time_series(x_fix_concat_rec, y_fix_concat_rec, k_fix_concat_rec, os.path.join(self.sample_dir, f'{step:06d}_{prefix}_rec.png'))
        print(f'Saved {prefix} time series samples into {self.sample_dir}...')

    def get_fixed_time_series(self, loader):
        """Get fixed time series with 2 observations from each class."""
        x_fix_list = []
        y_fix_list = []
        k_fix_list = []
        class_counts = {i: 0 for i in range(self.num_classes)}
        for x, y, k in loader:
            for i in range(x.size(0)):
                label = y[i].item()
                if class_counts[label] < 2:
                    x_fix_list.append(x[i])
                    y_fix_list.append(y[i])
                    k_fix_list.append(k[i])
                    class_counts[label] += 1
                if all(count == 2 for count in class_counts.values()):
                    return torch.stack(x_fix_list).to(self.device), torch.stack(y_fix_list).to(self.device), torch.stack(k_fix_list).to(self.device)
        return torch.stack(x_fix_list).to(self.device), torch.stack(y_fix_list).to(self.device), torch.stack(k_fix_list).to(self.device)

    def compute_gan_loss(self, out, target):
        """Compute GAN loss."""
        target_tensor = torch.ones_like(out) if target else torch.zeros_like(out)
        if self.loss_type == 'gan':
            return F.binary_cross_entropy_with_logits(out, target_tensor)
        elif self.loss_type == 'lsgan':
            return F.mse_loss(out, target_tensor)
        elif self.loss_type == 'wgan-gp':
            return -torch.mean(out) if target == 1 else torch.mean(out)

    def random_rotation_matrix(self):
        """Generate a random rotation matrix from a predefined set of quaternions."""

        # Randomly generate a quaternion
        q = np.random.rand(4)

        # Convert quaternion to rotation matrix
        q = torch.tensor(q, device=self.device, dtype=torch.float32)
        q = q / torch.norm(q)  # Normalize quaternion
        q0, q1, q2, q3 = q

        R = torch.tensor([
            [1 - 2*q2**2 - 2*q3**2, 2*q1*q2 - 2*q3*q0, 2*q1*q3 + 2*q2*q0],
            [2*q1*q2 + 2*q3*q0, 1 - 2*q1**2 - 2*q3**2, 2*q2*q3 - 2*q1*q0],
            [2*q1*q3 - 2*q2*q0, 2*q2*q3 + 2*q1*q0, 1 - 2*q1**2 - 2*q2**2]
        ], device=self.device, dtype=torch.float32)

        return R

    def augment_batch(self, x_real):
        """Apply random augmentations to the batch of real time series."""
        # Rotation
        min_val, max_val = -19.61, 19.61
        x_real_r = x_real * (max_val - min_val) + min_val  # De-normalize
        R = self.random_rotation_matrix()
        x_real_r = torch.matmul(R, x_real_r)  # Apply rotation
        x_real_r = (x_real_r - min_val) / (max_val - min_val)  # Re-normalize
        # # Scaling
        # alpha = 0.4 * torch.rand(1, device=self.device) + 0.8
        # x_real_r = alpha * x_real_r
        # # Translation
        # beta = 0.2 * torch.rand(1, device=self.device) - 0.1
        # x_real_r = x_real_r + beta
        # # Jittering
        # x_real_r = x_real_r + 0.01 * torch.randn_like(x_real_r)
        # # Clipping
        # x_real_r = torch.clamp(x_real_r, 0, 1)
        return x_real_r

    def train(self):
        """Train StarGAN within a single dataset."""
        # Set data loaders.
        train_loader = self.train_loader
        test_loader = self.test_loader
        td_loader = self.td_loader

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Fetch fixed inputs.
        x_train_fix, y_train_fix, k_train_fix = self.get_fixed_time_series(train_loader)
        x_train_fix = x_train_fix.to(self.device)
        y_train_fix = y_train_fix.to(self.device)
        k_train_fix = k_train_fix.to(self.device)
        x_test_fix, y_test_fix, k_test_fix = self.get_fixed_time_series(test_loader)
        x_test_fix = x_test_fix.to(self.device)
        y_test_fix = y_test_fix.to(self.device)
        k_test_fix = k_test_fix.to(self.device)

        # Get the last elapsed time from the log file
        initial_elapsed_time = self.get_last_elapsed_time()

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):

            # Update lambda_dom linearly
            lambda_dom = self.lambda_dom * (i / self.num_iters)

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real time series and labels.
            try:
                x_real, y_src, k_src = next(data_iter)
            except:
                data_iter = iter(train_loader)
                x_real, y_src, k_src = next(data_iter)

            # Fetch target domain time series. 
            try:
                x_td, y_td = next(data_iter_td)
            except:
                data_iter_td = iter(td_loader)
                x_td, y_td = next(data_iter_td)

            # Generate target class labels randomly.
            rand_idx = torch.randperm(y_src.size(0))
            y_trg = y_src[rand_idx]

            y_src_oh = self.label2onehot(y_src, self.num_classes)
            y_trg_oh = self.label2onehot(y_trg, self.num_classes)

            x_real = x_real.to(self.device)
            y_src = y_src.to(self.device)
            y_trg = y_trg.to(self.device)
            y_src_oh = y_src_oh.to(self.device)
            y_trg_oh = y_trg_oh.to(self.device)
            k_src = k_src.to(self.device)
            x_td = x_td.to(self.device)
            y_td = y_td.to(self.device)

            if self.augment:
                x_real_r = self.augment_batch(x_real)
            else:
                x_real_r = x_real
                
            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real time series.
            out_src, out_cls, out_dom, _ = self.D(x_real_r)
            d_loss_real = self.compute_gan_loss(out_src, 1)
            d_loss_cls = F.cross_entropy(out_cls, y_src)
            d_loss_dom = F.cross_entropy(out_dom, k_src)

            # Compute loss with fake time series.
            x_fake = self.G(x_real_r, y_trg_oh)
            out_src, _, _, _ = self.D(x_fake.detach())
            d_loss_fake = self.compute_gan_loss(out_src, 0)

            # Compute loss with real target domain time series.
            _, _, _, out_td = self.D(x_td)
            d_loss_td_real = self.compute_gan_loss(out_td, 1)

            # Compute loss with fake target domain time series.
            x_fake_td = self.G(x_td, y_trg_oh)
            _, _, _, out_td = self.D(x_fake_td.detach())
            d_loss_td_fake = self.compute_gan_loss(out_td, 0)

            # Compute loss for gradient penalty.
            alpha = torch.rand(x_real_r.size(0), 1, 1).to(self.device)
            x_hat = (alpha * x_real_r.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            out_src, _, _, _ = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)
            # x_td_hat = (alpha * x_td.data + (1 - alpha) * x_fake_td.data).requires_grad_(True)
            # _, _, _, out_td = self.D(x_td_hat)
            # d_loss_gp = self.gradient_penalty(out_td, x_td_hat)

            # Backward and optimize.
            d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp + lambda_dom * d_loss_dom
            # d_loss = self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp + lambda_dom * d_loss_dom + d_loss_td_real + d_loss_td_fake
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_cls'] = d_loss_cls.item()
            loss['D/loss_dom'] = d_loss_dom.item()
            loss['D/loss_gp'] = d_loss_gp.item()
            loss['D/loss_td_real'] = d_loss_td_real.item()
            loss['D/loss_td_fake'] = d_loss_td_fake.item()
            
            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #
            
            if (i+1) % self.n_critic == 0:
                # Original-to-target class.
                x_fake = self.G(x_real_r, y_trg_oh)
                out_src, out_cls, out_dom, _ = self.D(x_fake)
                g_loss_fake = self.compute_gan_loss(out_src, 1)
                g_loss_cls = F.cross_entropy(out_cls, y_trg)
                g_loss_dom = F.cross_entropy(out_dom, k_src)

                # Target-to-original class.
                x_reconst = self.G(x_fake, y_src_oh)
                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                # Original-to-original class.
                x_real_r_map = self.G(x_real_r, y_src_oh)
                g_loss_rot = torch.mean(torch.abs(x_real_r - x_real_r_map))

                # Compute loss with fake target domain time series.
                x_fake_td = self.G(x_td, y_trg_oh)
                _, _, _, out_td = self.D(x_fake_td)
                g_loss_td_fake = self.compute_gan_loss(out_td, 1)

                # Backward and optimize.
                g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls + lambda_dom * g_loss_dom + self.lambda_rot * g_loss_rot
                # g_loss = self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls + lambda_dom * g_loss_dom + self.lambda_rot * g_loss_rot + g_loss_td_fake
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_cls'] = g_loss_cls.item()
                loss['G/loss_dom'] = g_loss_dom.item()
                loss['G/loss_rot'] = g_loss_rot.item()
                loss['G/loss_td_fake'] = g_loss_td_fake.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                self.log_training_info(start_time, i, loss, initial_elapsed_time)

            # Sample time series.
            if (i+1) % self.sample_step == 0:
                self.sample_time_series(x_real_r, y_src, k_src, i+1, 'train')
                self.sample_time_series(x_train_fix, y_train_fix, k_train_fix, i+1, 'trainfix')
                self.sample_time_series(x_test_fix, y_test_fix, k_test_fix, i+1, 'testfix')

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Run evaluation.
            if (i+1) % self.eval_step == 0:
                run_evaluation(i+1, self.G, self.args)

            # Decay learning rates.
            if self.lr_update_step != -1:
                if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                    g_lr = max(g_lr - self.g_lr_step, self.min_g_lr)
                    d_lr = max(d_lr - self.d_lr_step, self.min_d_lr)
                    self.update_lr(g_lr, d_lr)
                    print('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))


    def custom_train_test_split(self, x, y, k, n_obs):
        # Combine y and k to identify unique combinations
        yk_comb = np.array(list(zip(y, k)))
        
        # Find unique combinations of y and k
        unique_combinations = np.unique(yk_comb, axis=0)
        
        # Initialize lists to hold train and test indices
        train_indices = []
        test_indices = []
        
        # Iterate over each unique combination
        for comb in unique_combinations:
            # Find indices of the current combination
            comb_indices = np.where((yk_comb == comb).all(axis=1))[0]
            
            # Shuffle indices to ensure randomness
            np.random.shuffle(comb_indices)
            
            # Select n_obs indices for the test set
            test_indices.extend(comb_indices[:n_obs])
            
            # Remaining indices go to the train set
            train_indices.extend(comb_indices[n_obs:])
        
        # Convert lists to numpy arrays
        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)
        
        # Split the data into train and test sets
        x_train, x_test = x[train_indices], x[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        k_train, k_test = k[train_indices], k[test_indices]
        
        return x_train, x_test, y_train, y_test, k_train, k_test

            

    @torch.no_grad()
    def sample(self, name):
        """Sample time series using the trained generator."""
        print('Start sampling...')
        # Load the trained generator.
        self.restore_model(self.resume_iters)

        with open(f'data/{self.dataset}.pkl', 'rb') as f:
            x, y, k = pickle.load(f)
        print(f'Loaded full dataset with shape {x.shape}, from {len(set(k))} domains and {len(set(y))} classes')

        with open(f'data/splits/{self.dataset}_df.pkl', 'rb') as f:
            x_df, y_df, k_df = pickle.load(f)
        print(f'Loaded Df data with shape {x_df.shape}, from {len(set(k_df))} domains and {len(set(y_df))} classes')

        with open(f'data/splits/{self.dataset}_dp.pkl', 'rb') as f:
            x_dp, y_dp, k_dp = pickle.load(f)
        print(f'Loaded Dp data with shape {x_dp.shape}, from {len(set(k_dp))} domains and {len(set(y_dp))} classes')

        with open(f'data/splits/{self.dataset}_dp_map.pkl', 'rb') as f:
            x_dp_map, y_dp_map, k_dp_map = pickle.load(f)
        print(f'Divided Dp data into map with shape {x_dp_map.shape}, from {len(set(k_dp_map))} domains and {len(set(y_dp_map))} classes')

        with open(f'data/splits/{self.dataset}_dp_te.pkl', 'rb') as f:
            x_dp_te, y_dp_te, k_dp_te = pickle.load(f)
        print(f'And into test with shape {x_dp_te.shape}, from {len(set(k_dp_te))} domains and {len(set(y_dp_te))} classes')

        # Create tensors
        x_dp_map = torch.tensor(x_dp_map, dtype=torch.float32, device=self.device)
        k_dp_map = torch.tensor(k_dp_map, dtype=torch.long, device=self.device)
        y_dp_map = torch.tensor(y_dp_map, dtype=torch.long, device=self.device)

        # Map x to the target classes
        x_syn, y_syn, k_syn = [], [], []
        for y_trg in range(0, self.num_classes):
            print(f'Mapping to class {y_trg}...')
            y_trg_tensor = torch.tensor([y_trg] * x_dp_map.size(0), device=self.device)
            y_trg_oh = self.label2onehot(y_trg_tensor, self.num_classes)
            x_syn.append(self.G(x_dp_map, y_trg_oh))
            y_syn.append(y_trg_tensor)
            k_syn.append(k_dp_map)
            
        x_syn = torch.cat(x_syn, dim=0).detach().cpu().numpy()
        y_syn = torch.cat(y_syn, dim=0).detach().cpu().numpy()
        k_syn = torch.cat(k_syn, dim=0).detach().cpu().numpy()

        print(f'Loaded Syn data with shape {x_syn.shape}, from {len(set(k_syn))} domains and {len(set(y_syn))} classes')

        # Save the data
        with open(f'data/splits/{self.dataset}_{name}.pkl', 'wb') as f:
            pickle.dump((x_syn, y_syn, k_syn), f)

            

    @torch.no_grad()
    def sample_wal(self, name):
        """Sample time series using the trained generator."""
        print('Start sampling...')
        # Load the trained generator.
        self.restore_model(self.resume_iters)

        with open(f'data/{self.dataset}.pkl', 'rb') as f:
            x, y, k = pickle.load(f)
        print(f'Loaded full dataset with shape {x.shape}, from {len(set(k))} domains and {len(set(y))} classes')

        with open(f'data/splits/{self.dataset}_df.pkl', 'rb') as f:
            x_df, y_df, k_df = pickle.load(f)
        print(f'Loaded Df data with shape {x_df.shape}, from {len(set(k_df))} domains and {len(set(y_df))} classes')

        with open(f'data/splits/{self.dataset}_dp.pkl', 'rb') as f:
            x_dp, y_dp, k_dp = pickle.load(f)
        print(f'Loaded Dp data with shape {x_dp.shape}, from {len(set(k_dp))} domains and {len(set(y_dp))} classes')

        with open(f'data/splits/{self.dataset}_dp_map.pkl', 'rb') as f:
            x_dp_map, y_dp_map, k_dp_map = pickle.load(f)
        print(f'Divided Dp data into map with shape {x_dp_map.shape}, from {len(set(k_dp_map))} domains and {len(set(y_dp_map))} classes')

        with open(f'data/splits/{self.dataset}_dp_te.pkl', 'rb') as f:
            x_dp_te, y_dp_te, k_dp_te = pickle.load(f)
        print(f'And into test with shape {x_dp_te.shape}, from {len(set(k_dp_te))} domains and {len(set(y_dp_te))} classes')

        mask_dp_map_wal = y_dp_map == 0
        x_dp_map_wal, y_dp_map_wal, k_dp_map_wal = x_dp_map[mask_dp_map_wal], y_dp_map[mask_dp_map_wal], k_dp_map[mask_dp_map_wal]
        print(f'Extracted from Dp map data wal data with shape {x_dp_map_wal.shape}, from {len(set(k_dp_map_wal))} domains and {len(set(y_dp_map_wal))} classes')

        # Create tensors
        x_dp_map = torch.tensor(x_dp_map_wal, dtype=torch.float32, device=self.device)
        k_dp_map = torch.tensor(k_dp_map_wal, dtype=torch.long, device=self.device)
        y_dp_map = torch.tensor(y_dp_map_wal, dtype=torch.long, device=self.device)

        # Map x to the target classes
        x_syn, y_syn, k_syn = [], [], []
        for y_trg in range(0, self.num_classes):
            print(f'Mapping to class {y_trg}...')
            y_trg_tensor = torch.tensor([y_trg] * x_dp_map.size(0), device=self.device)
            y_trg_oh = self.label2onehot(y_trg_tensor, self.num_classes)
            x_syn.append(self.G(x_dp_map, y_trg_oh))
            y_syn.append(y_trg_tensor)
            k_syn.append(k_dp_map)
            
        x_syn = torch.cat(x_syn, dim=0).detach().cpu().numpy()
        y_syn = torch.cat(y_syn, dim=0).detach().cpu().numpy()
        k_syn = torch.cat(k_syn, dim=0).detach().cpu().numpy()

        print(f'Loaded Syn data with shape {x_syn.shape}, from {len(set(k_syn))} domains and {len(set(y_syn))} classes')

        # Save the data
        with open(f'data/splits/{self.dataset}_{name}.pkl', 'wb') as f:
            pickle.dump((x_syn, y_syn, k_syn), f)





