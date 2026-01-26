import torch
import numpy as np
from torch_geometric.data import Data
from config import GlobalParams
import importlib
from dataclasses import dataclass
import os
import data.dataset_path
import data.dataset_branch
import data.dataset_path_mask
import data.dataset_branch_mask

def get_datset(params):
    seed=params.seed
    #ファイルによってnpzファイル指定
    save_dir="data/cache"
    os.makedirs(save_dir,exist_ok=True)
    save_path=os.path.join(save_dir,f"data_{params.load_file}_seed_{seed}_mask{params.mask_ratio}.npz")

    if os.path.exists(save_path):
        print(f"-- Loading dataset (seed:{seed}) --")
        return np.load(save_path,allow_pickle=True)
        
    print(f">>>No dataset : seed{seed} -> make new dataset")
    if (params.load_file=="data.dataset_path"):
        result=data.dataset_path.generate_data(params.seed)
    elif (params.load_file=="data.dataset_branch"):
        result=data.dataset_branch.generate_data(params.seed)
    elif (params.load_file=="data.dataset_path_mask"):
        result=data.dataset_path_mask.generate_data(params.seed,params.mask_ratio)
    elif (params.load_file=="data.dataset_branch_mask"):
        result=data.dataset_branch_mask.generate_data(params.seed,params.mask_ratio) 
        
    np.savez_compressed(save_path,**result)
    return np.load(save_path,allow_pickle=True)

def create_torch_data_list(ds,params,model):
    #--dataset作成--
    data_list=[]

    dataset_dict=ds['dataset'].item()
    volt_source_dict=ds['voltge_source'].item()
    source_node_dict=ds['source_node_array'].item()

    for i in range(params.num_data):
        x=torch.tensor(dataset_dict[i].T,dtype=torch.float32)
        y=torch.tensor(volt_source_dict[i],dtype=torch.float32)
        target_node_index=torch.tensor([source_node_dict[i]],dtype=torch.long)

        data=Data(x=x,y=y.unsqueeze(0),edge_index=model.union_edge_index,target_idx=target_node_index)
        data_list.append(data)
    return data_list
