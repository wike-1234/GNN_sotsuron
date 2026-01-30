#!/bin/bash
python3 train_myGCN.py --load_file data.dataset_path --seed 5 --mask_ratio 1
python3 train_myGCN.py --load_file data.dataset_path --seed 17 --mask_ratio 1
python3 train_myGCN.py --load_file data.dataset_path --seed 42 --mask_ratio 1
python3 train_myGCN.py --load_file data.dataset_path --seed 70 --mask_ratio 1
python3 train_myGCN.py --load_file data.dataset_path --seed 100 --mask_ratio 1

python3 train_normalGCN.py --load_file data.dataset_path --seed 5 --mask_ratio 1
python3 train_normalGCN.py --load_file data.dataset_path --seed 17 --mask_ratio 1
python3 train_normalGCN.py --load_file data.dataset_path --seed 42 --mask_ratio 1
python3 train_normalGCN.py --load_file data.dataset_path --seed 70 --mask_ratio 1
python3 train_normalGCN.py --load_file data.dataset_path --seed 100 --mask_ratio 1

python3 train_myGCN.py --load_file data.dataset_branch --seed 5 --mask_ratio 1
python3 train_myGCN.py --load_file data.dataset_branch --seed 17 --mask_ratio 1
python3 train_myGCN.py --load_file data.dataset_branch --seed 42 --mask_ratio 1
python3 train_myGCN.py --load_file data.dataset_branch --seed 70 --mask_ratio 1
python3 train_myGCN.py --load_file data.dataset_branch --seed 100 --mask_ratio 1

python3 train_normalGCN.py --load_file data.dataset_branch --seed 5 --mask_ratio 1
python3 train_normalGCN.py --load_file data.dataset_branch --seed 17 --mask_ratio 1
python3 train_normalGCN.py --load_file data.dataset_branch --seed 42 --mask_ratio 1
python3 train_normalGCN.py --load_file data.dataset_branch --seed 70 --mask_ratio 1
python3 train_normalGCN.py --load_file data.dataset_branch --seed 100 --mask_ratio 1

python3 train_myGCN.py --load_file data.dataset_path_mask --seed 5 --mask_ratio 2
python3 train_myGCN.py --load_file data.dataset_path_mask --seed 17 --mask_ratio 2
python3 train_myGCN.py --load_file data.dataset_path_mask --seed 42 --mask_ratio 2
python3 train_myGCN.py --load_file data.dataset_path_mask --seed 70 --mask_ratio 2
python3 train_myGCN.py --load_file data.dataset_path_mask --seed 100 --mask_ratio 2