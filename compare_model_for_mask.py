import torch
import importlib
import torch.nn as nn
import random
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from dataclasses import dataclass
from torch_geometric.nn import MessagePassing, global_add_pool
from torch_geometric.utils import softmax


from config import GlobalParams
from utils.graph_utils import hop_index, channel_edge_index,set_seed
from models.GNN_model import AttentionGCN
import dataset

#--ファイル指定--
npz_file_1="data_data.dataset_branch_seed_42_mask1.npz"
npz_file_2="data_data.dataset_branch_mask_seed_42_mask2.npz"
pth_file_1="pth_branch/model_dataset_branch_myGCN_seed42_mask1.pth"
pth_file_2="pth_branch_mask2/model_dataset_branch_mask_myGCN_seed42_mask2.pth"
seed=42

#loadするnpz指定
save_dir="data/cache"
save_path=os.path.join(save_dir,npz_file_1)
ds_1=np.load(save_path,allow_pickle=True)
save_path=os.path.join(save_dir,npz_file_2)
ds_2=np.load(save_path,allow_pickle=True)

@dataclass
class Superparams_1:
    out_channels=GlobalParams.out_channels
    lr=GlobalParams.lr
    num_epoch=GlobalParams.num_epoch
    batch_size=GlobalParams.batch_size
    train_ratio=GlobalParams.train_ratio
    lambda_balance=GlobalParams.lambda_balance
    mask_ratio=GlobalParams.mask_ratio
    if ("mask2" in npz_file_1) or ("mask5" in npz_file_1) or ("mask10" in npz_file_1):
        in_channels=2*int(ds_1['data_step'])
    else:
        in_channels=int(ds_1['data_step'])
    volt_step=int(ds_1['volt_step'])
    num_nodes=int(ds_1['num_nodes'])
    num_data=int(ds_1['num_data'])
    B=ds_1['B']

@dataclass
class Superparams_2:
    out_channels=GlobalParams.out_channels
    lr=GlobalParams.lr
    num_epoch=GlobalParams.num_epoch
    batch_size=GlobalParams.batch_size
    train_ratio=GlobalParams.train_ratio
    lambda_balance=GlobalParams.lambda_balance
    mask_ratio=GlobalParams.mask_ratio
    if ("mask2" in npz_file_2) or ("mask5" in npz_file_2) or ("mask10" in npz_file_2):
        in_channels=2*int(ds_2['data_step'])
    else:
        in_channels=int(ds_2['data_step'])
    volt_step=int(ds_2['volt_step'])
    num_nodes=int(ds_2['num_nodes'])
    num_data=int(ds_2['num_data'])
    B=ds_2['B']

#--parameter--
params_1=Superparams_1()
params_2=Superparams_2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
#originalの部分
#特徴量毎のグラフ定義
#kステップ目のデータ-k-hop目までつながった行列で定義
structure_dict_1={}
if "my" in pth_file_1:
    if ("mask2" in npz_file_1) or ("mask5" in npz_file_1) or ("mask10" in npz_file_1):
        for node in range(params_1.in_channels):
            structure_dict_1[node]=hop_index(int(node/2)+1,params_1)
    else:
        for node in range(params_1.in_channels):
            structure_dict_1[node]=hop_index(node+1,params_1)

elif "normal" in pth_file_1:
    common_hop=hop_index(1,params_1)
    for node in range(params_1.in_channels):
        structure_dict_1[node]=common_hop    

union_index_1,union_mask_1=channel_edge_index(params_1,structure_dict_1)

structure_dict_2={}
if "my" in pth_file_2:
    if ("mask2" in npz_file_2) or ("mask5" in npz_file_2) or ("mask10" in npz_file_2):
        for node in range(params_2.in_channels):
            structure_dict_2[node]=hop_index(int(node/2)+1,params_2)
    else:
        for node in range(params_2.in_channels):
            structure_dict_2[node]=hop_index(node+1,params_2)

elif "normal" in pth_file_2:
    common_hop=hop_index(1,params_2)
    for node in range(params_2.in_channels):
        structure_dict_2[node]=common_hop    

union_index_2,union_mask_2=channel_edge_index(params_2,structure_dict_2)


#--model定義--
model_A=AttentionGCN(params_1,union_index_1,union_mask_1)
model_B=AttentionGCN(params_2,union_index_2,union_mask_2)
model_A=model_A.to(device)
model_B=model_B.to(device)
optimizer=torch.optim.Adam(model_A.parameters(),lr=params_1.lr,weight_decay=1e-4)
criterion_class=nn.CrossEntropyLoss()
criterion_reg=nn.MSELoss()

#data list作成
data_list_1=dataset.create_torch_data_list(ds_1,params_1,model_A)
#--train/test--
indices=list(range(params_1.num_data))
random.shuffle(indices)

train_size=int(params_1.num_data*params_1.train_ratio)
train_indices=indices[:train_size]
test_indices=indices[train_size:]

train_data=[data_list_1[i] for i in train_indices]
test_data=[data_list_1[i] for i in test_indices]

train_loader_1=DataLoader(train_data,batch_size=params_1.batch_size,shuffle=True)
test_loader_1=DataLoader(test_data,batch_size=params_1.batch_size,shuffle=True)

data_list_2=dataset.create_torch_data_list(ds_2,params_2,model_B)
#--train/test--
indices=list(range(params_2.num_data))
random.shuffle(indices)

train_size=int(params_2.num_data*params_2.train_ratio)
train_indices=indices[:train_size]
test_indices=indices[train_size:]

train_data=[data_list_2[i] for i in train_indices]
test_data=[data_list_2[i] for i in test_indices]

train_loader_2=DataLoader(train_data,batch_size=params_2.batch_size,shuffle=True)
test_loader_2=DataLoader(test_data,batch_size=params_2.batch_size,shuffle=True)

#modelのロード
model_A.load_state_dict(torch.load(pth_file_1,map_location=torch.device('cpu')))
model_B.load_state_dict(torch.load(pth_file_2,map_location=torch.device('cpu')))
print("load completed")

# === att_linの重みを比較 ===
weight_A=model_A.att_lin.weight.detach().cpu().numpy().flatten()
weight_B=model_B.att_lin.weight.detach().cpu().numpy().flatten()

fig, axes=plt.subplots(1,2,figsize=(16,6))
x_A=np.arange(1,params_1.in_channels+1)
axes[0].bar(x_A,weight_A)
axes[0].set_title("Case A : Attention Linear Weight")
axes[0].set_xlabel("Channel")
axes[0].set_ylabel("Weight")

x_B=np.arange(1,params_2.in_channels+1)
axes[1].bar(x_B,weight_B)
axes[1].set_title("Case B : Attention Linear Weight")
axes[1].set_xlabel("Channel")
axes[1].set_ylabel("Weight")

plt.show()

# === 出力値の分布を可視化 ===
test_batch_1=next(iter(test_loader_1))
test_batch_1=test_batch_1.to(device)
test_batch_2=next(iter(test_loader_2))
test_batch_2=test_batch_2.to(device)
with torch.no_grad():
    pred_A,_=model_A(test_batch_1.x, test_batch_1.edge_index, test_batch_1.batch,params_1)
    pred_B,_=model_B(test_batch_2.x, test_batch_2.edge_index, test_batch_2.batch,params_2)

out_A=pred_A.cpu().view(-1).numpy()
out_B=pred_B.cpu().view(-1).numpy()

threshold=0.1

mask=((np.abs(out_A) > threshold) | (np.abs(out_B) > threshold))


plt.figure(figsize=(6,6))
plt.scatter(out_A[mask],out_B[mask],alpha=0.5,s=20)
plt.plot([out_A.min(),out_A.max()], [out_A.min(),out_A.max()],'r--')
plt.xlabel("Case A Predoction")
plt.ylabel("Case B Prediction")
plt.title("Prediction Comparison")
plt.grid()
plt.show()