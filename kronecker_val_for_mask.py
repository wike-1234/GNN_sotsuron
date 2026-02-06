import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import torch
import torch.nn as nn

from config import GlobalParams
from utils.graph_utils import set_seed,hop_index,channel_edge_index
from models.GNN_model import AttentionGCN


plt.rcParams.update({
    'font.size':11,
    'axes.labelsize':11,
    'xtick.labelsize':10,
    'ytick.labelsize':10,
    'legend.fontsize':10
})

#file指定
pth_file="pth_branch_mask10/model_dataset_branch_mask_myGCN_seed42_mask10.pth"
npz_file="data_data.dataset_branch_mask_seed_42_mask5.npz"
seed=42

#dataが多すぎるのでinput_nodeを絞る
target_input_node=0
target_output_node=1

#loadするnpz指定
save_dir="data/cache"
save_path=os.path.join(save_dir,npz_file)
ds=np.load(save_path,allow_pickle=True)

@dataclass
class Superparams:
    out_channels=GlobalParams.out_channels
    lr=GlobalParams.lr
    num_epoch=GlobalParams.num_epoch
    batch_size=GlobalParams.batch_size
    train_ratio=GlobalParams.train_ratio
    lambda_balance=GlobalParams.lambda_balance
    mask_ratio=GlobalParams.mask_ratio
    if ("mask2" in npz_file) or ("mask5" in npz_file) or ("mask10" in npz_file):
        in_channels=2*int(ds['data_step'])
    else:
        in_channels=int(ds['data_step'])
    volt_step=int(ds['volt_step'])
    num_nodes=int(ds['num_nodes'])
    num_data=int(ds['num_data'])
    B=ds['B']

set_seed(seed)
params=Superparams()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

structure_dict={}
if "my" in pth_file:
    if ("mask2" in npz_file) or ("mask5" in npz_file) or ("mask10" in npz_file):
        for node in range(params.in_channels):
            structure_dict[node]=hop_index(int(node/2)+1,params)
    else:
        for node in range(params.in_channels):
            structure_dict[node]=hop_index(node+1,params)

elif "normal" in pth_file:
    common_hop=hop_index(1,params)
    for node in range(params.in_channels):
        structure_dict[node]=common_hop    

union_index,union_mask=channel_edge_index(params,structure_dict)

#--model定義--
model=AttentionGCN(params,union_index,union_mask)
model=model.to(device)
optimizer=torch.optim.Adam(model.parameters(),lr=params.lr,weight_decay=1e-4)
criterion_class=nn.CrossEntropyLoss()
criterion_reg=nn.MSELoss()

#modelのロード
model.load_state_dict(torch.load(pth_file,map_location=torch.device('cpu')))
print("load completed")

model.eval()

#隣接行列
Adjacency_matrix={}
for ch in range(params.in_channels):
    with torch.no_grad():
        w=model.edge_mask*model.edge_weight_base
        ch_weight=w[:,ch].cpu().numpy()
    edge_index=model.union_edge_index.cpu().numpy()
    source=edge_index[0]
    distination=edge_index[1]
    #隣接行列の枠を作成
    A=np.zeros((params.num_nodes,params.num_nodes))
    #Aにedge_weightを対応させる
    for i in range((len(ch_weight))):
        u=source[i]
        v=distination[i]
        A[u,v]=ch_weight[i]
    Adjacency_matrix[ch]=A

#Attention Linear
with torch.no_grad():
    W=model.val_lin.weight.cpu().numpy()

N=params.num_nodes
K=params.in_channels
V=params.volt_step

A_stack=np.array([Adjacency_matrix[ch] for ch in range(params.in_channels)])
Interaction_Tensor=np.einsum('kji,ok -> iojk',A_stack,W)
Op=Interaction_Tensor.reshape(N*params.volt_step,N*K)



start_col=target_input_node*K
end_col=(target_input_node+1)*K
start_row=target_output_node*V
end_row=(target_output_node+1)*V
Op_focused_all=Op[start_row:end_row,start_col:end_col]
Op_focused=Op_focused_all[:,::2]

max_op=np.max(np.abs(Op_focused))
Op_normalized=Op_focused/max_op



H, W_mat = Op_normalized.shape
ratios = [params.in_channels, 1, 2, 1]
fig, axes = plt.subplots(figsize=(6,5))



sns.heatmap(Op_normalized,
            ax=axes,
            cmap="RdBu_r",
            center=0,
            vmax=1,vmin=-1,
            cbar=True,
            square=False,
            linewidths=0.0,
            linecolor='lightgray',
            cbar_kws={"label":"Coefficient Values (Normalized)"})

if "mask2" in pth_file:
    case=1
elif "mask5" in pth_file:
    case=2
elif "mask10" in pth_file:
    case=3
# タイトルと軸ラベル
axes.set_title(f"GNN Linear Operator -Value- (Case{case})")
axes.set_ylabel(f"Output Feature (Node:{target_output_node+1})")
axes.set_xlabel(f"Input Feature (Node:{target_input_node+1})")

step = 10 
all_ticks = np.arange(W_mat)
all_labels = [k+1 for k in range(W_mat)]
axes.set_xticks(all_ticks[::step]+0.5)
axes.set_xticklabels(all_labels[::step], rotation=0)

all_ticks = np.arange(H)
all_labels = [k+1 for k in range(H)]
axes.set_yticks(all_ticks[::step]+0.5)
axes.set_yticklabels(all_labels[::step], rotation=0)

plt.tight_layout()
plt.show()