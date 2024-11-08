from core.model import Generator, Discriminator, DomainClassifier
from core.eval import run_evaluation
from core.augment import augment_data
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
import csv
import matplotlib.pyplot as plt


class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, train_loader, test_loader, args):
        """Initialize configurations."""

        self.args = args

        # Data loader.
        self.train_loader = train_loader
        self.test_loader = test_loader

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

        # Learning rate decay.
        self.total_decay_steps = (self.num_iters - self.num_iters_decay) // self.lr_update_step
        self.g_lr_step = self.g_lr / self.total_decay_steps
        self.d_lr_step = self.d_lr / self.total_decay_steps

        # Miscellaneous.
        self.mode = args.mode
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.aumgent = args.augment

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
        self.log_file = os.path.join(self.log_dir, 'log.csv')
        file_exists = os.path.isfile(self.log_file)
        with open(self.log_file, 'a', newline='') as csvfile:
            fieldnames = ['Elapsed Time', 'Iteration'] + [f'D/{key}' for key in ['loss_real', 'loss_fake', 'loss_cls', 'loss_gp']] + [f'G/{key}' for key in ['loss_fake', 'loss_rec', 'loss_cls']]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()

        # Build the model.
        self.build_model()

    def build_model(self):
        """Create a generator and a discriminator."""
        self.G = Generator(self.g_conv_dim, self.num_classes, self.g_repeat_num)
        self.D = Discriminator(self.num_timesteps, self.d_conv_dim, self.num_classes, self.d_repeat_num)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')
            
        self.G.to(self.device)
        self.D.to(self.device)

        # # Load the pretrained domain classifier
        # print('Loading the pretrained domain classifier...')
        # self.domain_classifier_df = DomainClassifier(self.num_channels, self.num_df_domains, self.num_classes, self.num_timesteps)
        # # self.domain_classifier_df.load_state_dict(torch.load(f'pretrained_nets/domain_classifier_{self.dataset}_df.ckpt', map_location=self.device, weights_only=False))
        # self.domain_classifier_df.load_state_dict(torch.load(f'pretrained_nets/domain_classifier_{self.dataset}_df.ckpt', map_location=self.device))
        # self.domain_classifier_df.eval()
        # self.domain_classifier_df.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        # print(model)
        # print(name)
        # print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
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

    def save_time_series(self, data, labels, filename):
        N = data.size(0)
        ncols = 2 * self.num_classes
        nrows = N // ncols
        fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 2.5), sharex=True, sharey=True)
        axs = axs.flatten()
        for idx in range(N):
            for i in range(data.size(1)):
                axs[idx].plot(data[idx, i, :].detach().cpu().numpy(), label=self.channel_names[i], linewidth=0.7)
            # axs[idx].set_ylim(0, 1)
            axs[idx].axis('off')
            axs[idx].set_title(f'{self.class_names[labels[idx].item()]}')
            if idx < ncols:
                axs[idx].legend(loc='lower left')
        for idx in range(N, len(axs)):
            axs[idx].axis('off')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    @torch.no_grad()
    def sample_time_series(self, x_fix, y_fix, step, prefix):
        """Sample and save time series."""
        x_fix_list_map = [x_fix]
        y_fix_list_map = [y_fix]
        x_fix_list_rec = [x_fix]
        y_fix_list_rec = [y_fix]
        for y in range(self.num_classes):
            # Map original class to target class.
            y_trg = torch.tensor([y] * x_fix.size(0)).to(self.device)
            y_trg_oh = self.label2onehot(y_trg, self.num_classes)
            x_fix_list_map.append(self.G(x_fix, y_trg_oh.to(self.device)))
            y_fix_list_map.append(y_trg)
            # Map target class to original class.
            y_src_oh = self.label2onehot(y_fix, self.num_classes)
            x_fix_list_rec.append(self.G(x_fix, y_src_oh.to(self.device)))
            y_fix_list_rec.append(y_fix)
        x_fix_concat_map = torch.cat(x_fix_list_map, dim=0)
        y_fix_concat_map = torch.cat(y_fix_list_map, dim=0)
        x_fix_concat_rec = torch.cat(x_fix_list_rec, dim=0)
        y_fix_concat_rec = torch.cat(y_fix_list_rec, dim=0)
        self.save_time_series(x_fix_concat_map, y_fix_concat_map, os.path.join(self.sample_dir, f'{step:06d}_{prefix}_map.png'))
        self.save_time_series(x_fix_concat_rec, y_fix_concat_rec, os.path.join(self.sample_dir, f'{step:06d}_{prefix}_rec.png'))
        print(f'Saved {prefix} time series samples into {self.sample_dir}...')

    def get_fixed_time_series(self, loader):
        """Get fixed time series with 2 observations from each class."""
        x_fix_list = []
        y_fix_list = []
        class_counts = {i: 0 for i in range(self.num_classes)}
        for x, y, k in loader:
            for i in range(x.size(0)):
                label = y[i].item()
                if class_counts[label] < 2:
                    x_fix_list.append(x[i])
                    y_fix_list.append(y[i])
                    class_counts[label] += 1
                if all(count == 2 for count in class_counts.values()):
                    return torch.stack(x_fix_list).to(self.device), torch.stack(y_fix_list).to(self.device)
        return torch.stack(x_fix_list).to(self.device), torch.stack(y_fix_list).to(self.device)
    
    def augment_data(self, data):
        B, M, T = data.shape
        augmented_data = torch.zeros_like(data).to(self.device)

        for i in range(B):
            acc = data[i, :, :]

            # Random channel swapping
            permuted_indices = torch.randperm(M)  # Get a random permutation of channels
            acc = acc[permuted_indices, :]

            # Independent mirroring for each channel
            for j in range(M):
                if torch.rand(1).item() > 0.5:  # 50% chance to mirror each channel independently
                    acc[j, :] = 1 - acc[j, :]  # Mirror by flipping around y = 0.5

            # Assign to augmented_data
            augmented_data[i, :, :] = acc

        return augmented_data

    def train(self):
        """Train StarGAN within a single dataset."""
        # Set data loaders.
        train_loader = self.train_loader
        test_loader = self.test_loader

        # Fetch fixed inputs.
        x_train_fix, y_train_fix = self.get_fixed_time_series(train_loader)
        x_train_fix = x_train_fix.to(self.device)
        y_train_fix = y_train_fix.to(self.device)
        x_test_fix, y_test_fix = self.get_fixed_time_series(test_loader)
        x_test_fix = x_test_fix.to(self.device)
        y_test_fix = y_test_fix.to(self.device)

        self.sample_time_series(x_train_fix, y_train_fix, 0, 'train')
        self.sample_time_series(x_test_fix, y_test_fix, 0, 'test')

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Get the last elapsed time from the log file
        initial_elapsed_time = self.get_last_elapsed_time()

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real time series and labels.
            try:
                x_real, y_src, k_src = next(data_iter)
            except:
                data_iter = iter(train_loader)
                x_real, y_src, k_src = next(data_iter)
            
            # Augment the data
            if self.aumgent:
                x_real = self.augment_data(x_real)

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

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real time series.
            out_src, out_cls = self.D(x_real)
            d_loss_real = - torch.mean(out_src)
            d_loss_cls = F.cross_entropy(out_cls, y_src)

            # Compute loss with fake time series.
            x_fake = self.G(x_real, y_trg_oh)
            out_src, out_cls = self.D(x_fake.detach())
            d_loss_fake = torch.mean(out_src)

            # Compute loss for gradient penalty.
            alpha = torch.rand(x_real.size(0), 1, 1).to(self.device)
            x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            out_src, _ = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            # Backward and optimize.
            d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_cls'] = d_loss_cls.item()
            loss['D/loss_gp'] = d_loss_gp.item()
            
            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #
            
            if (i+1) % self.n_critic == 0:
                # Original-to-target class.
                x_fake = self.G(x_real, y_trg_oh)
                out_src, out_cls = self.D(x_fake)
                g_loss_fake = - torch.mean(out_src)
                g_loss_cls = F.cross_entropy(out_cls, y_trg)

                # Target-to-original class.
                x_reconst = self.G(x_fake, y_src_oh)
                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                # # Domain classification loss.
                # out_dom = self.domain_classifier_df(x_fake, y_trg)
                # g_loss_dom = F.cross_entropy(out_dom, k_src)

                # Backward and optimize.
                # g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls + self.lambda_dom * g_loss_dom
                g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_cls'] = g_loss_cls.item()
                # loss['G/loss_dom'] = g_loss_dom.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                self.log_training_info(start_time, i, loss, initial_elapsed_time)

            # Sample time series.
            if (i+1) % self.sample_step == 0:
                self.sample_time_series(x_train_fix, y_train_fix, i+1, 'train')
                self.sample_time_series(x_test_fix, y_test_fix, i+1, 'test')

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
                    g_lr -= self.g_lr_step
                    d_lr -= self.d_lr_step
                    self.update_lr(g_lr, d_lr)
                    print('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))
