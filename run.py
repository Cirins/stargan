import subprocess

mode = 'train'

class_names = ['WAL', 'RUN', 'CLD', 'CLU']
channel_names = ['X', 'Y', 'Z']



dataset = 'pamap_mobiact'



if dataset == 'realworld_mobiact':
    num_df_domains = 15
    num_dp_domains = 61

elif dataset == 'mobiact_realworld':
    num_df_domains = 61
    num_dp_domains = 15

elif dataset == 'realworld_pamap':
    num_df_domains = 15
    num_dp_domains = 6

elif dataset == 'pamap_realworld':
    num_df_domains = 6
    num_dp_domains = 15

elif dataset == 'mobiact_pamap':
    num_df_domains = 61
    num_dp_domains = 6

elif dataset == 'pamap_mobiact':
    num_df_domains = 6
    num_dp_domains = 61



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
                '--log_step', str(log_step),
                '--sample_step', str(sample_step),
                '--model_save_step', str(model_save_step),
                '--eval_step', str(eval_step),
                # '--resume_iters', str(resume_iters),
                ])

