import subprocess

class_names = ['WAL', 'RUN', 'CLD', 'CLU']
channel_names = ['X', 'Y', 'Z']
num_df_domains = 15
num_dp_domains = 61

dataset = 'realworld_mobiact'

mode = 'train'

g_lr = 1e-4
d_lr = 1e-4

lambda_dom = 0
lambda_rot = 10

augment = True

log_step = 100
sample_step = 1000
model_save_step = 10000
eval_step = 10000

resume_iters = 0



subprocess.run(['python', 'main.py',
                '--mode', mode,
                '--dataset', dataset,
                '--class_names', ' '.join(class_names),
                '--channel_names', ' '.join(channel_names),
                '--num_df_domains', str(num_df_domains),
                '--num_dp_domains', str(num_dp_domains),
                '--g_lr', str(g_lr),
                '--d_lr', str(d_lr),
                '--lambda_dom', str(lambda_dom),
                '--lambda_rot', str(lambda_rot),
                '--log_step', str(log_step),
                '--sample_step', str(sample_step),
                '--model_save_step', str(model_save_step),
                '--eval_step', str(eval_step),
                '--augment', str(augment),
                # '--resume_iters', str(resume_iters),
                ])

