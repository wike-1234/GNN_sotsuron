import torch
import numpy as np
from torch_geometric.data import Data
from config import GlobalParams
import importlib
from dataclasses import dataclass 


ds=importlib.import_module(GlobalParams.load_file)

@dataclass
class Hyperparams:
    out_channels=GlobalParams.out_channels
    lr=GlobalParams.lr
    num_epoch=GlobalParams.num_epoch
    batch_size=GlobalParams.batch_size
    train_ratio=GlobalParams.train_ratio
    lambda_balance=GlobalParams.lambda_balance
    load_file=GlobalParams.load_file
    seed=GlobalParams.seed
    if (GlobalParams.load_file=="data.dataset_path") or(GlobalParams.load_file=="data.dataset_branch"):
        in_channels=ds.data_step
    elif (GlobalParams.load_file=="data.dataset_path_mask") or(GlobalParams.load_file=="data.dataset_branch_mask"):
        in_channels=2*ds.data_step
    volt_step=ds.volt_step
    num_nodes=ds.num_node
    num_data=ds.num_data
    B=ds.B

def create_torch_data_list(ds,params,model):
    #--dataset作成--
    data_list=[]
    for i in range(params.num_data):
        x=torch.tensor(ds.dataset[i].T,dtype=torch.float32)
        y=torch.tensor(ds.voltage_source[i],dtype=torch.float32)
        target_node_index=torch.tensor([ds.source_node_array[i]],dtype=torch.long)

        data=Data(x=x,y=y.unsqueeze(0),edge_index=model.union_edge_index,target_idx=target_node_index)
        data_list.append(data)
    return data_list
