# NBDI: A Simple and Efficient Termination Condition for Skill Extraction from Task-Agnostic Demonstrations

This is the PyTorch implementation of the paper "**NBDI: A Simple and Efficient Termination Condition for Skill Extraction from Task-Agnostic Demonstrations**".  
This provides NBDI algorithm working in `kitchen` environment.

## Requirements

- python 3.8+
- mujoco 2.0 (for RL experiments)
- Ubuntu 18.04

## Getting Started
Create a virtual environment and install all required packages.
```
cd nbdi
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```
Set the environment variables that specify the root experiment and data directories. For example:
```
mkdir ./experiments
mkdir ./data
export EXP_DIR=./experiments
export DATA_DIR=./data
```
Install the [D4RL benchmark](https://github.com/kpertsch/d4rl) repository by following its installation instructions.

### Train Commands
To train a skill prior model for the kitchen environment, run:
```
python train.py --path nbdi/configs/skill_prior_learning/kitchen/hierarchical_cl --val_data_size 160 --gpu 0
```
For training a NBDI agent on the kitchen environment using the pre-trained skill prior from above, run:
```
python train_rl.py --path nbdi/configs/hrl/kitchen/nbdi_cl --seed 0 --gpu 0
```

