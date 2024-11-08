import subprocess

dataset = 'realworld'
class_names = ['WAL', 'RUN', 'CLD', 'CLU']
channel_names = ['X', 'Y', 'Z']
num_df_domains = 10
num_dp_domains = 15



subprocess.run(['python', 'main.py',
                '--mode', 'train',
                '--dataset', dataset,
                '--class_names', ' '.join(class_names),
                '--channel_names', ' '.join(channel_names),
                '--num_df_domains', str(num_df_domains),
                '--num_dp_domains', str(num_dp_domains)
                ])

