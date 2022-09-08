import os

os.system(f"python train.py --method=fast_naive --t_len=100 --beta_requires_grad=True --readout_max=False --single_spike=True --dataset=yinyang --load_spatial_dims=False --use_augmentation=False --epoch=200 --batch=128 --lr=0.001 --track_activity=True")
os.system(f"python train.py --method=fast_naive --t_len=100 --beta_requires_grad=True --readout_max=False --single_spike=True --dataset=mnist --load_spatial_dims=False --use_augmentation=True --epoch=140 --batch=128 --lr=0.001 --track_activity=True")
os.system(f"python train.py --method=fast_naive --t_len=500 --beta_requires_grad=True --readout_max=False --single_spike=True --dataset=shd --load_spatial_dims=False --use_augmentation=False --epoch=200 --batch=128 --lr=0.0002 --track_activity=True")