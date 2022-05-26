import os
import sys
sys.path.append("/home/luketaylor/PycharmProjects/BrainBox")
sys.path.append("/home/luketaylor/PycharmProjects/FastSNN")
sys.path.append("/data/dpag-auditory-neuroscience/kebl6283/PycharmProjects/BrainBox")
sys.path.append("/data/dpag-auditory-neuroscience/kebl6283/PycharmProjects/FastSNN")

n_repeats = 3
path = "/data/dpag-auditory-neuroscience/kebl6283/PycharmProjects/FastSNN"
n_hidden = 200
skip_connections = True
dt = 1

fast_layer_list = [False, True]
dataset = "fmnist"
epoch = 150
batch_size = 128
lr = 0.0002
n_layers = 1

for fast_layer in fast_layer_list:
    for _ in range(n_repeats):
        os.system(f"bash create_job.sh --path={path} --n_hidden={n_hidden} --n_layers={n_layers} --fast_layer={fast_layer} --skip_connections={skip_connections} --dt={dt} --dataset={dataset} --epoch={epoch} --batch_size={batch_size} --lr={lr}")