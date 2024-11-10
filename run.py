import subprocess

dataset = 'realworld'
class_names = ['WAL', 'RUN', 'CLD', 'CLU']
channel_names = ['X', 'Y', 'Z']
num_df_domains = 10
num_dp_domains = 5

mode = 'finetune'

g_lr = 1e-6
d_lr = 1e-6
lambda_dom = 1

log_step = 100
sample_step = 1000
model_save_step = 10000
eval_step = 10000

resume_iters = 370000



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
                '--log_step', str(log_step),
                '--sample_step', str(sample_step),
                '--model_save_step', str(model_save_step),
                '--eval_step', str(eval_step),
                '--resume_iters', str(resume_iters),
                ])

