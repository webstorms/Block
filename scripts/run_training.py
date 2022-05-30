import os


def launch_training(dataset, dt, epoch, lr, batch_size=128, n_hidden=200, n_repeats=6):
    for fast_layer in [False, True]:
        for _ in range(n_repeats):
            os.system(f"python train.py --n_hidden={n_hidden} --fast_layer={fast_layer} --dt={dt} --dataset={dataset} --epoch={epoch} --batch_size={batch_size} --lr={lr}")
            # Will likely rather want to sbatch jobs onto a GPU cluster


launch_training("fmnist", dt=1, epoch=150, lr=0.0002)
launch_training("nmnist", dt=1, epoch=150, lr=0.0002)
launch_training("shd", dt=2, epoch=200, lr=0.001)
