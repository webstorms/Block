import os


os.system(f"bash create_job.sh yin --method=fast_naive --t_len=100 --beta_requires_grad=True --readout_max=False --single_spike=True --dataset=yinyang --load_spatial_dims=False --use_augmentation=False --epoch=200 --batch=128 --lr=0.001")
#os.system(f"bash create_job.sh yin --method=standard --t_len=100 --beta_requires_grad=False --readout_max=False --single_spike=True --dataset=yinyang --load_spatial_dims=False --use_augmentation=False --epoch=200 --batch=128 --lr=0.001")
#os.system(f"bash create_job.sh yin --method=standard --t_len=100 --beta_requires_grad=False --readout_max=False --single_spike=False --dataset=yinyang --load_spatial_dims=False --use_augmentation=False --epoch=200 --batch=128 --lr=0.001")

#os.system(f"bash create_job.sh shd --method=fast_naive --t_len=500 --beta_requires_grad=True --readout_max=False --single_spike=True --dataset=shd --load_spatial_dims=False --use_augmentation=False --epoch=200 --batch=128 --lr=0.0002")
#os.system(f"bash create_job.sh cmnist --method=fast_naive --t_len=8 --beta_requires_grad=True --readout_max=True --single_spike=True --dataset=mnist --load_spatial_dims=True --use_augmentation=True --epoch=140 --batch=64 --lr=0.001")

# Yin-Yang
#os.system(f"bash create_job.sh yin --method=standard --t_len=100 --beta_requires_grad=True --readout_max=False --single_spike=True --dataset=yinyang --load_spatial_dims=False --use_augmentation=False --epoch=200 --batch=128 --lr=0.001")
#os.system(f"bash create_job.sh yin --method=standard --t_len=100 --beta_requires_grad=True --readout_max=False --single_spike=False --dataset=yinyang --load_spatial_dims=False --use_augmentation=False --epoch=200 --batch=128 --lr=0.001")

# MNIST
# os.system(f"bash create_job.sh cmnist --method=standard --t_len=8 --beta_requires_grad=True --readout_max=True --single_spike=True --dataset=mnist --load_spatial_dims=True --use_augmentation=True --epoch=140 --batch=64 --lr=0.001")
# os.system(f"bash create_job.sh cmnist --method=standard --t_len=8 --beta_requires_grad=True --readout_max=True --single_spike=False --dataset=mnist --load_spatial_dims=True --use_augmentation=True --epoch=140 --batch=64 --lr=0.001")
# os.system(f"bash create_job.sh lmnist --method=standard --t_len=100 --beta_requires_grad=True --readout_max=False --single_spike=True --dataset=mnist --load_spatial_dims=False --use_augmentation=True --epoch=140 --batch=128 --lr=0.001")
# os.system(f"bash create_job.sh lmnist --method=standard --t_len=100 --beta_requires_grad=True --readout_max=False --single_spike=False --dataset=mnist --load_spatial_dims=False --use_augmentation=True --epoch=140 --batch=128 --lr=0.001")
#os.system(f"bash create_job.sh lmnist --method=fast_naive --t_len=100 --beta_requires_grad=True --readout_max=False --single_spike=True --dataset=mnist --load_spatial_dims=False --use_augmentation=True --epoch=140 --batch=128 --lr=0.001")

# F-MNIST
#os.system(f"bash create_job.sh cfmnist --method=standard --t_len=8 --beta_requires_grad=True --readout_max=False --single_spike=True --dataset=fmnist --load_spatial_dims=True --use_augmentation=True --epoch=140 --batch=64 --lr=0.001")
#os.system(f"bash create_job.sh cfmnist --method=standard --t_len=8 --beta_requires_grad=True --readout_max=False --single_spike=False --dataset=fmnist --load_spatial_dims=True --use_augmentation=True --epoch=140 --batch=64 --lr=0.001")
#os.system(f"bash create_job.sh lfmnist --method=standard --t_len=100 --beta_requires_grad=True --readout_max=False --single_spike=True --dataset=fmnist --load_spatial_dims=False --use_augmentation=True --epoch=140 --batch=128 --lr=0.001")
#os.system(f"bash create_job.sh lfmnist --method=standard --t_len=100 --beta_requires_grad=True --readout_max=False --single_spike=False --dataset=fmnist --load_spatial_dims=False --use_augmentation=True --epoch=140 --batch=128 --lr=0.001")

# N-MNIST
#os.system(f"bash create_job.sh nmnist --method=standard --t_len=300 --beta_requires_grad=True --readout_max=False --single_spike=True --dataset=nmnist --load_spatial_dims=False --use_augmentation=False --epoch=200 --batch=128 --lr=0.0002")
#os.system(f"bash create_job.sh nmnist --method=standard --t_len=300 --beta_requires_grad=True --readout_max=False --single_spike=False --dataset=nmnist --load_spatial_dims=False --use_augmentation=False --epoch=200 --batch=128 --lr=0.0002")

# SHD
#os.system(f"bash create_job.sh shd --method=standard --t_len=500 --beta_requires_grad=True --readout_max=False --single_spike=True --dataset=shd --load_spatial_dims=False --use_augmentation=False --epoch=200 --batch=128 --lr=0.0002")
#os.system(f"bash create_job.sh shd --method=standard --t_len=500 --beta_requires_grad=True --readout_max=False --single_spike=False --dataset=shd --load_spatial_dims=False --use_augmentation=False --epoch=200 --batch=128 --lr=0.0002")









# Yin-Yang
# os.system(f"bash create_job.sh yin --method=fast_naive --t_len=100 --beta_requires_grad=True --readout_max=False --single_spike=False --dataset=yinyang --load_spatial_dims=False --use_augmentation=False --epoch=200 --batch=128 --lr=0.001")
# os.system(f"bash create_job.sh standyin train.py --method=standard --t_len=100 --single_spike=False --dataset=yinyang --load_spatial_dims=False --use_augmentation=False --epoch=200 --batch=128 --lr=0.001")
# os.system(f"bash create_job.sh standyin train.py --method=standard --t_len=100 --single_spike=True --dataset=yinyang --load_spatial_dims=False --use_augmentation=False --epoch=200 --batch=128 --lr=0.001")
#
# # MNIST and FMNIST
# for beta_requires_grad in [True, False]:
#    for readout_max in [True, False]:
#        os.system(f"bash create_job.sh lmnist --method=fast_naive --t_len=100 --beta_requires_grad={beta_requires_grad} --readout_max={readout_max} --single_spike=True --dataset=mnist --load_spatial_dims=False --use_augmentation=True --epoch=140 --batch=128 --lr=0.001")
#
# for beta_requires_grad in [True, False]:
#     for readout_max in [True, False]:
#         os.system(f"bash create_job.sh lfmnist --method=fast_naive --t_len=100 --beta_requires_grad={beta_requires_grad} --readout_max={readout_max} --single_spike=True --dataset=fmnist --load_spatial_dims=False --use_augmentation=True --epoch=140 --batch=128 --lr=0.001")
#
# for beta_requires_grad in [True, False]:
#     for readout_max in [True, False]:
#         os.system(f"bash create_job.sh cmnist --method=fast_naive --t_len=8 --beta_requires_grad={beta_requires_grad} --readout_max={readout_max} --single_spike=True --dataset=mnist --load_spatial_dims=True --use_augmentation=True --epoch=140 --batch=64 --lr=0.001")
#
# for beta_requires_grad in [True, False]:
#     for readout_max in [True, False]:
#         os.system(f"bash create_job.sh cfmnist --method=fast_naive --t_len=8 --beta_requires_grad={beta_requires_grad} --readout_max={readout_max} --single_spike=True --dataset=fmnist --load_spatial_dims=True --use_augmentation=True --epoch=140 --batch=64 --lr=0.001")
#
# # N-MNIST
# os.system(f"bash create_job.sh nmnist --method=fast_naive --t_len=300 --beta_requires_grad=False --readout_max=False --single_spike=True --dataset=nmnist --load_spatial_dims=False --use_augmentation=False --epoch=200 --batch=128 --lr=0.0002")
# os.system(f"bash create_job.sh nmnist --method=fast_naive --t_len=300 --beta_requires_grad=True --readout_max=False --single_spike=True --dataset=nmnist --load_spatial_dims=False --use_augmentation=False --epoch=200 --batch=128 --lr=0.0002")
#
# # SHD
# os.system(f"bash create_job.sh shd --method=fast_naive --t_len=500 --beta_requires_grad=False --readout_max=False --single_spike=True --dataset=shd --load_spatial_dims=False --use_augmentation=False --epoch=200 --batch=128 --lr=0.0002")
# os.system(f"bash create_job.sh shd --method=fast_naive --t_len=500 --beta_requires_grad=True --readout_max=False --single_spike=True --dataset=shd --load_spatial_dims=False --use_augmentation=False --epoch=200 --batch=128 --lr=0.0002")
