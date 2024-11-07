import subprocess

dataset = 'realworld_mobiact'
class_names = ['WAL', 'RUN', 'CLD', 'CLU']
channel_names = ['X', 'Y', 'Z']
num_df_domains = 15
num_dp_domains = 61
lambda_dom = 0
# lambda_rec = 100
# augment = True
resume_iters = 70000



subprocess.run(['python', 'main.py',
                '--mode', 'finetune',
                '--dataset', 'realworld_mobiact',
                '--class_names', ' '.join(class_names),
                '--channel_names', ' '.join(channel_names),
                '--num_df_domains', str(num_df_domains),
                '--num_dp_domains', str(num_dp_domains),
                '--lambda_dom', str(lambda_dom),
                # '--lambda_rec', str(lambda_rec),
                # '--augment', str(augment),
                '--resume_iters', str(resume_iters),
                ])

