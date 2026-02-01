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
npz_file_1="data_data.dataset_path_seed_42_mask1.npz"
npz_file_2="data_data.dataset_path_mask_seed_42_mask2.npz"
pth_file_1="pth_path/model_dataset_path_myGCN_seed42_mask1.pth"
pth_file_2="pth_path_mask2/model_dataset_path_mask_myGCN_seed42_mask2.pth"
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
    for node in range(params_1.in_channels):
        structure_dict_1[node]=hop_index(node+1,params_1)
elif "normal" in pth_file_1:
    common_hop=hop_index(1,params_1)
    for node in range(params_1.in_channels):
        structure_dict_1[node]=common_hop    

union_index,union_mask=channel_edge_index(params_1,structure_dict_1)

structure_dict_2={}
if "my" in pth_file_2:
    for node in range(params_2.in_channels):
        structure_dict_2[node]=hop_index(node+2,params_1)
elif "normal" in pth_file_1:
    common_hop=hop_index(1,params_1)
    for node in range(params_1.in_channels):
        structure_dict_1[node]=common_hop    

union_index,union_mask=channel_edge_index(params_1,structure_dict_1)


#--model定義--
model_A=AttentionGCN(params,union_index,union_mask)
model_B=AttentionGCN(params,union_index,union_mask)
model_A=model_A.to(device)
model_B=model_B.to(device)
optimizer=torch.optim.Adam(model_A.parameters(),lr=params.lr,weight_decay=1e-4)
criterion_class=nn.CrossEntropyLoss()
criterion_reg=nn.MSELoss()

#data list作成
data_list=dataset.create_torch_data_list(ds,params,model_A)
#--train/test--
indices=list(range(params.num_data))
random.shuffle(indices)

train_size=int(params.num_data*params.train_ratio)
train_indices=indices[:train_size]
test_indices=indices[train_size:]

train_data=[data_list[i] for i in train_indices]
test_data=[data_list[i] for i in test_indices]

train_loader=DataLoader(train_data,batch_size=params.batch_size,shuffle=True)
test_loader=DataLoader(test_data,batch_size=params.batch_size,shuffle=True)


#modelのロード
model_A.load_state_dict(torch.load(pth_file_1,map_location=torch.device('cpu')))
model_B.load_state_dict(torch.load(pth_file_2,map_location=torch.device('cpu')))
print("load completed")

# === att_linの重みを比較 ===
weight_A=model_A.att_lin.weight.detach().cpu().numpy().flatten()
weight_B=model_B.att_lin.weight.detach().cpu().numpy().flatten()

fig, axes=plt.subplots(1,2,figsize=(16,6))
x=np.arange(1,params.in_channels+1)
axes[0].bar(x,weight_A)
axes[0].set_title("Case A : Attention Linear Weight")
axes[0].set_xlabel("Channel")
axes[0].set_ylabel("Weight")

axes[1].bar(x,weight_B)
axes[1].set_title("Case B : Attention Linear Weight")
axes[1].set_xlabel("Channel")
axes[1].set_ylabel("Weight")

plt.show()

# === 出力値の分布を可視化 ===
test_batch=next(iter(test_loader))
test_batch=test_batch.to(device)
with torch.no_grad():
    pred_A,_=model_A(test_batch.x, test_batch.edge_index, test_batch.batch,params)
    pred_B,_=model_B(test_batch.x, test_batch.edge_index, test_batch.batch,params)

out_A=pred_A.cpu().view(-1).numpy()
out_B=pred_B.cpu().view(-1).numpy()

threshold=0.1

mask=((np.abs(out_A) > threshold) | (np.abs(out_B) > threshold))


plt.figure(figsize=(6,6))
plt.scatter(out_A[mask],out_B[mask],alpha=0.5,s=20)
plt.plot([out_A.min(),out_A.max()], [out_A.min(),out_A.max()],'r--')
plt.xlabel("Model A Predoction")
plt.ylabel("Model B Prediction")
plt.title("Prediction Comparison")
plt.grid()
plt.show()