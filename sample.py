import subprocess

class_names = ['WAL', 'RUN', 'CLD', 'CLU']
channel_names = ['X', 'Y', 'Z']
num_df_domains = 15
num_dp_domains = 61

dataset = 'realworld_mobiact'

mode = 'sample'

syn_name = 'rwma07'

resume_iters = 200000



subprocess.run(['python', 'main.py',
                '--mode', mode,
                '--dataset', dataset,
                '--class_names', ' '.join(class_names),
                '--channel_names', ' '.join(channel_names),
                '--num_df_domains', str(num_df_domains),
                '--num_dp_domains', str(num_dp_domains),
                '--syn_name', syn_name,
                '--resume_iters', str(resume_iters),
                ])

