#!/bin/bash

# step
CUDA_VISIBLE_DEVICES=1 python train.py --dataset step --batch_size 30 --num_replicate 10 --low_layers '[1, 2, 1]' --high_layers '[1, 2, 2]'
CUDA_VISIBLE_DEVICES=1 python train.py --dataset step --batch_size 30 --num_replicate 10 --low_layers '[1, 2, 2, 1]' --high_layers '[1, 2, 2, 2]'