#!/bin/bash

python train.py --path=/home/luketaylor/PycharmProjects/FastSNN --n_hidden=300 --n_layers=5 --fast_layer=True --skip_connections=True --dt=2 --dataset=shd --epoch=100 --batch_size=128 --lr=0.001