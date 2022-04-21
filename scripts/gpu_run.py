import os
import torch
import numpy as np

#project_path = '/data/dpag-auditory-neuroscience/kebl6283/PycharmProjects/FastSNN'
project_path = '/home/luketaylor/PycharmProjects/FastSNN'

n_epochs = 150
batch_size = 256
lr = 0.0002

layer_list = [0, 1, 2]
hidden_list = ['[200,200]', '[200]']
t_list = [100]

beta_init = np.exp(-1/10)

os.system(f'python train.py --path={project_path} --layer_type=0 --n_per_hidden=[200,200] --t_len=100 --beta_init={beta_init} --bias_diff=False --dataset_cash=True --epoch={n_epochs} --batch_size={batch_size} --lr={lr}')
os.system(f'python train.py --path={project_path} --layer_type=0 --n_per_hidden=[200] --t_len=100 --beta_init={beta_init} --bias_diff=False --dataset_cash=True --epoch={n_epochs} --batch_size={batch_size} --lr={lr}')

for hidden in hidden_list:
    for t in t_list:
        for layer in layer_list:
            os.system(f'python train.py --path={project_path} --layer_type={layer} --n_per_hidden={hidden} --t_len={t} --beta_init={beta_init} --bias_diff=True --dataset_cash=True --epoch={n_epochs} --batch_size={batch_size} --lr={lr}')
            #os.system(f'bash create_job.sh {layer}:{hidden}:{t} --path={project_path} --layer_type={layer} --n_per_hidden={hidden} --t_len={t} --dataset_cash=False --epoch={n_epochs} --batch_size={batch_size} --lr={lr}')

