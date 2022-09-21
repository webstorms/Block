import os
#
# n = 1
# r = True
# d_list = [1, 2, 4, 8, 16, 32]
# l_list = [1, 2, 3]
#
# for d in d_list:
#     for l in l_list:
#         os.system(f"python train.py --method=fast_naive --t_len=500 --beta_requires_grad=True --readout_max=False --single_spike=True --d={d} --recurrent={r} --n_layers={l} --detach_recurrent_spikes=True --dataset=shd --load_spatial_dims=False --use_augmentation=False --epoch=120 --batch=128 --lr=0.001")

os.system(f"python train.py --method=standard --t_len=500 --beta_requires_grad=True --readout_max=False --single_spike=False --recurrent=True --n_layers=1 --detach_recurrent_spikes=True --dataset=shd --load_spatial_dims=False --use_augmentation=False --epoch=120 --batch=128 --lr=0.001")