# Deep Gaussian Process Models for Integrating Multifidelity Experiments with Non-stationary Relationships

This repository contains code for the paper
**"Deep Gaussian Process Models for Integrating Multifidelity Experiments with Non-stationary Relationships"**
by Jongwoo Ko and Heeyoung Kim.

## Requirements
- This codebase is written for `python3`.
- To install necessary python packages, run `pip install -r requirements.txt`.

## Training
```
python train.py --dataset step --num_replicate 10 --low_layer '[1, 2, 1]' --high_layer '[1, 2, 2]'
```

### Arguments
```
python train.py [-h] [--dataset] [--batch_size] [--num_replicate] [--low_layer] [--high_layer]
                [--low_learning_rate] [--high_learning_rate] [--low_epoch] [--high_epoch]
```