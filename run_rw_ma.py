import subprocess

class_names = ['WAL', 'RUN', 'CLD', 'CLU']
channel_names = ['X', 'Y', 'Z']
num_df_domains = 15
num_dp_domains = 61

dataset = 'realworld_mobiact'

mode = 'train'

lambda_cls = 1
lambda_rec = 10
lambda_gp = 10
lambda_dom = 0
lambda_rot = 10

g_repeat_num = 5
d_repeat_num = 4

batch_size = 32

augment = True

g_lr = 1e-4
d_lr = 1e-4
min_g_lr = 1e-6
min_d_lr = 1e-6
num_iters = 400000
num_iters_decay = 200000
lr_update_step = -1 # -1 means no lr update

log_step = 100
sample_step = 1000
model_save_step = 10000
eval_step = 10000

# resume_iters = 0



subprocess.run(['python', 'main.py',
                '--mode', mode,
                '--dataset', dataset,
                '--class_names', ' '.join(class_names),
                '--channel_names', ' '.join(channel_names),
                '--num_df_domains', str(num_df_domains),
                '--num_dp_domains', str(num_dp_domains),
                '--lambda_cls', str(lambda_cls),
                '--lambda_rec', str(lambda_rec),
                '--lambda_gp', str(lambda_gp),
                '--lambda_dom', str(lambda_dom),
                '--lambda_rot', str(lambda_rot),
                '--g_repeat_num', str(g_repeat_num),
                '--d_repeat_num', str(d_repeat_num),
                '--batch_size', str(batch_size),
                '--augment', str(augment),
                '--g_lr', str(g_lr),
                '--d_lr', str(d_lr),
                '--min_g_lr', str(min_g_lr),
                '--min_d_lr', str(min_d_lr),
                '--num_iters', str(num_iters),
                '--num_iters_decay', str(num_iters_decay),
                '--lr_update_step', str(lr_update_step),
                '--log_step', str(log_step),
                '--sample_step', str(sample_step),
                '--model_save_step', str(model_save_step),
                '--eval_step', str(eval_step),
                # '--resume_iters', str(resume_iters),
                ])

